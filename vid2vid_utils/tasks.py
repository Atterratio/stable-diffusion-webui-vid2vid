import gc
import json
import shutil
from collections import Counter
from re import compile as Regex
from time import time
from traceback import format_exc
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import Compose

from tqdm import tqdm

from modules.deepbooru import model as deepbooru_model
from modules.devices import torch_gc, device, autocast, cpu
from modules.shared import state
from repositories.midas.midas.dpt_depth import DPTDepthModel
from repositories.midas.midas.midas_net import MidasNet
from repositories.midas.midas.midas_net_custom import MidasNet_small
from repositories.midas.midas.transforms import NormalizeImage, Resize, PrepareForNet
from vid2vid_utils.constants import *
from vid2vid_utils.enums import ExtractFrame, MidasModel
from vid2vid_utils.external_tools import FFMPEG_BIN, MIDAS_MODEL_PATH, CURL_BIN, RESR_BIN, RIFE_BIN
from vid2vid_utils.img_utils import get_im, im_shift_01, im_to_img, dtype, get_img
from vid2vid_utils.utils import *


TaskResponse = Tuple[RetCode, str]

def task(fn: Callable[..., TaskResponse]):
    def wrapper(*args, **kwargs):
        global cur_task
        task_name = fn.__name__[5:]  # remove '_btn_'
        if workspace is None:
            code = RetCode.ERROR
            info = 'no current workspace opened!'
            ts = None
        elif cur_task is not None:
            code = RetCode.ERROR
            info = f'task {cur_task!r} is stilling running!'
            ts = None
        else:
            cur_task = task_name
            print(f'>> run task {task_name!r}')
            state.interrupted = False
            _ts = time()
            code, info = fn(*args, **kwargs)
            ts = time() - _ts
            cur_task = None
        return gr_update_status(info, code=code, task=task_name, ts=ts)

    return wrapper


@task
def _btn_ffmpeg_extract(video_file: object, extract_frame: str, extract_fmt: str, extract_fps: float) -> TaskResponse:
    out_dp = workspace / WS_FRAMES
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('extract')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    out_fp = workspace / WS_AUDIO
    if out_fp.exists():
        out_fp.unlink()

    try:
        # ref:
        #   - https://ffmpeg.org/ffmpeg.html
        #   - https://zhuanlan.zhihu.com/p/85895180
        # ffprobe -i test.mp4 -v quiet -select_streams v -show_entries frame=pkt_pts_time,pict_type

        extract_frame: ExtractFrame = ExtractFrame(extract_frame)
        if extract_frame == ExtractFrame.FPS:
            cmd = f'"{FFMPEG_BIN}" -i "{video_file.name}" -an -sn -f image2 -q:v 2 -r {extract_fps} "{out_dp}{os.sep}%05d.{extract_fmt}"'
        elif extract_frame == ExtractFrame.IPB:
            cmd = f'"{FFMPEG_BIN}" -i "{video_file.name}" -an -sn -f image2 -q:v 2 -fps_mode vfr "{out_dp}{os.sep}%05d.{extract_fmt}"'
        else:  # I/P/B
            cmd = f'"{FFMPEG_BIN}" -i "{video_file.name}" -an -sn -f image2 -q:v 2 -fps_mode vfr -vf "select=eq(pict_type\,{extract_frame.name})" "{out_dp}{os.sep}%05d.{extract_fmt}"'
        sh(cmd)

        has_audio = 'no'
        if cur_process_audio:
            for stream in ffprob_info['streams']:
                if stream['codec_type'] == 'audio':
                    sh(f'"{FFMPEG_BIN}" -i "{video_file.name}" -vn -sn "{out_fp}"')
                    has_audio = 'yes'
                    break

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}, audio: {has_audio}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc()
        print(e)
        return RetCode.ERROR, e


@task
def _btn_frame_delta() -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    out_dp = workspace / WS_DFRAME
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('framedelta')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        fps = list(in_dp.iterdir())
        im0, im1 = None, get_im(fps[0], mode='RGB')
        for fp in tqdm(fps[1:]):
            if state.interrupted: break

            im0, im1 = im1, get_im(fp, mode='RGB')
            delta = im1 - im0  # [-1, 1]
            im = im_shift_01(delta)  # [0, 1]
            img = im_to_img(im)
            img.save(out_dp / f'{fp.stem}.png')

        return RetCode.INFO, f'framedelta: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc();
        print(e)
        return RetCode.ERROR, e
    finally:
        torch_gc()
        gc.collect()


@task
def _btn_midas(midas_model) -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    out_dp = workspace / WS_DEPTH
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('midas')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        urls = {
            MidasModel.DPT_LARGE: 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt',
            MidasModel.DPT_HYBRID: 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt',
            MidasModel.MIDAS_V21: 'https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt',
            MidasModel.MIDAS_V21_SMALL: 'https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt',
        }
        midas_model: MidasModel = MidasModel(midas_model)
        url = urls[midas_model]
        model_path = MIDAS_MODEL_PATH / Path(url).name
        if not model_path.exists():
            MIDAS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            sh(f'{CURL_BIN} {url} -L -C - -o "{model_path}"')

        if midas_model == MidasModel.DPT_LARGE:
            model = DPTDepthModel(path=model_path, backbone="vitl16_384", non_negative=True)
            net_w, net_h = 384, 384
            resize_mode = 'minimal'
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model == MidasModel.DPT_HYBRID:
            model = DPTDepthModel(path=model_path, backbone="vitb_rn50_384", non_negative=True)
            net_w, net_h = 384, 384
            resize_mode = 'minimal'
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model == MidasModel.MIDAS_V21:
            model = MidasNet(model_path, non_negative=True)
            net_w, net_h = 384, 384
            resize_mode = 'upper_bound'
            normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif midas_model == MidasModel.MIDAS_V21_SMALL:
            model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                                   non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
            resize_mode = 'upper_bound'
            normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        cuda = torch.device('cuda')
        model = model.to(device)
        if device == cuda: model = model.to(memory_format=torch.channels_last)
        model.eval()

        transform = Compose([
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ])

        with torch.no_grad(), autocast():
            for fn in tqdm(list(in_dp.iterdir())):  # TODO: make batch for speedup
                if state.interrupted: break

                im = get_im(fn, mode='RGB')  # [H, W, C], float32
                X_np = transform({'image': im})['image']  # [C, maxH, maxW], float32
                X = torch.from_numpy(X_np).to(device).unsqueeze(0)  # [B=1, C, maxH, maxW], float32
                if device == cuda: X = X.to(memory_format=torch.channels_last)

                pred = model.forward(X)

                depth = F.interpolate(pred.unsqueeze(1), size=im.shape[:2], mode='bicubic', align_corners=False)
                depth = depth.squeeze().cpu().numpy().astype(dtype)  # [H, W], float32
                vmin, vmax = depth.min(), depth.max()
                if vmax - vmin > np.finfo(depth.dtype).eps:
                    depth_n = (depth - vmin) / (vmax - vmin)
                else:
                    depth_n = np.zeros_like(depth)
                depth_n = np.expand_dims(depth_n, axis=-1)  # [H, W, C=1]

                img = im_to_img(depth_n)
                img.save(out_dp / f'{Path(fn).stem}.png')

        return RetCode.INFO, f'depth_masks: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc();
        print(e)
        return RetCode.ERROR, e
    finally:
        if model in locals(): del model
        torch_gc()
        gc.collect()


@task
def _btn_deepdanbooru(topk=32) -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    out_fp = workspace / WS_TAGS
    if out_fp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('deepdanbooru')
        out_fp.unlink()

    try:
        tags: Dict[str, str] = {}
        deepbooru_model.start()
        for fp in tqdm(sorted(in_dp.iterdir())):
            img = get_img(fp, mode='RGB')
            tags[fp.name] = deepbooru_model.tag_multi(img)

        tags_flatten = []
        for prompt in tags.values():
            tags_flatten.extend([t.strip() for t in prompt.split(',')])
        tags_topk = ', '.join([t for t, c in Counter(tags_flatten).most_common(topk)])

        with open(out_fp, 'w', encoding='utf-8') as fh:
            json.dump(tags, fh, indent=2, ensure_ascii=False)
        with open(workspace / WS_TAGS_TOPK, 'w', encoding='utf-8') as fh:
            fh.write(tags_topk)

        return RetCode.INFO, f'prompts: {len(tags)}, top-{topk} freq tags: {tags_topk}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc();
        print(e)
        return RetCode.ERROR, e
    finally:
        deepbooru_model.model.to(cpu)
        torch_gc()
        gc.collect()


@task
def _btn_resr(resr_model: str) -> TaskResponse:
    in_dp = workspace / WS_IMG2IMG
    if not in_dp.exists():
        return RetCode.ERROR, f'img2img folder not found: {in_dp}'

    out_dp = workspace / WS_RESR
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('resr')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        try:
            m = Regex('-x(\d)').search(resr_model).groups()
            resr_ratio = int(m[0])
        except:
            print('>> cannot parse `resr_ratio` form model name, defaults to 2')
            resr_ratio = 2

        sh(f'"{RESR_BIN}" -v -s {resr_ratio} -n {resr_model} -i "{in_dp}" -o "{out_dp}"')

        return RetCode.INFO, f'upscaled: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc();
        print(e)
        return RetCode.ERROR, e


@task
def _btn_rife(rife_model: str, rife_ratio: float, extract_fmt: str) -> TaskResponse:
    in_dp = workspace / WS_RESR
    if not in_dp.exists():
        return RetCode.ERROR, f'resr folder not found: {in_dp}'

    out_dp = workspace / WS_RIFE
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('rife')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        n_interp = int(get_folder_file_count(in_dp) * rife_ratio)
        sh(f'"{RIFE_BIN}" -v -n {n_interp} -m {rife_model} -f %05d.{extract_fmt} -i "{in_dp}" -o "{out_dp}"')

        return RetCode.INFO, f'interpolated: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc();
        print(e)
        return RetCode.ERROR, e


@task
def _btn_ffmpeg_export(export_fmt: str, frame_src: str, extract_fmt: str, extract_fps: float, extract_frame: str,
                       rife_ratio: float) -> TaskResponse:
    in_dp = workspace / frame_src
    if not in_dp.exists():
        return RetCode.ERROR, f'src folder not found: {in_dp}'

    audio_opts = ''
    in_fp = workspace / WS_AUDIO
    if cur_process_audio and in_fp.exists():
        audio_opts += f' -i "{in_fp}"'

    out_fp = workspace / f'{WS_SYNTH}-{frame_src}.{export_fmt}'
    if out_fp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('export')
        out_fp.unlink()

    def get_real_fps() -> float:
        real_fps = None

        n_frames = get_folder_file_count(in_dp)
        if real_fps is None:  # if video duration available
            try:
                for stream in ffprob_info['streams']:
                    if stream['codec_type'] == 'video':
                        real_fps = n_frames / float(stream['duration'])
                        break
            except:
                pass
        if real_fps is None:  # if extracted FPS is known
            if ExtractFrame(extract_frame) == ExtractFrame.FPS:
                real_fps = extract_fps * rife_ratio
        if real_fps is None:  # if video fps available
            try:
                n_inits = get_folder_file_count(workspace / WS_FRAMES)
                for stream in ffprob_info['streams']:
                    if stream['codec_type'] == 'video':
                        real_fps = float(stream['avg_frame_rate']) * n_frames / n_inits
                        break
            except:
                pass
        if real_fps is None:  # default
            print(f'cannot decide real fps, defaults to extract_fps: {extract_fps}')
            real_fps = extract_fps

        return real_fps

    def get_ext() -> str:
        exts = {os.path.splitext(fn)[-1] for fn in os.listdir(in_dp)}
        if len(exts) > 1:
            print(f'>> warn: found multiple file extensions in src foulder: {exts}')
        return list(exts)[0]

    try:
        try:
            sh(f'"{FFMPEG_BIN}"{audio_opts} -framerate {get_real_fps()} -i "{in_dp}{os.sep}%05d{get_ext()}" -crf 20 -c:v libx264 -pix_fmt yuv420p "{out_fp}"')
        except:
            sh(f'"{FFMPEG_BIN}"{audio_opts} -framerate {get_real_fps()} -i "{in_dp}{os.sep}%05d{get_ext()}" "{out_fp}"')

        return RetCode.INFO, f'filesize: {get_file_size(out_fp):.3f}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc();
        print(e)
        return RetCode.ERROR, e
