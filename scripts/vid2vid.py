import json
from traceback import format_exc
from modules.shared import state
import shutil

from scripts.prompt_travel import process_images_before, process_images_after  # type: ignore

from modules import scripts
from modules.images import resize_image
from modules.processing import Processed, StableDiffusionProcessingImg2Img, get_fixed_seed, process_images, \
    process_images_inner
from modules.script_callbacks import remove_callbacks_for_function, on_before_image_saved, ImageSaveParams, \
    on_cfg_denoiser, CFGDenoiserParams
from modules.sd_samplers_common import setup_img2img_steps
from modules.ui import gr_show
from vid2vid_utils.external_tools import *
from vid2vid_utils.img_utils import *
from vid2vid_utils.tasks import _btn_ffmpeg_extract, _btn_frame_delta,_btn_deepdanbooru, _btn_midas, _btn_resr, _btn_rife, _btn_ffmpeg_export
from vid2vid_utils.utils import *
from vid2vid_utils.constants import *


def _file_select(video_file: object) -> List[GradioRequest]:
    global cur_cache_folder, workspace, ffprob_info

    # close workspace
    if video_file is None:
        ws_name = workspace.name
        workspace = None
        ffprob_info = None

        return [
            gr.Text.update(label=LABEL_CACHE_FOLDER, value=cur_cache_folder, interactive=True),
            gr.TextArea.update(visible=False),
            gr_update_status(f'closed workspace {ws_name!r}'),
        ]

    # open existing workspace
    ws_dp = cur_cache_folder / get_workspace_name(video_file.orig_name)
    info_fp = Path(ws_dp) / WS_FFPROBE
    if ws_dp.exists():
        workspace = ws_dp
        ws_name = workspace.name

        with open(info_fp, 'r', encoding='utf-8') as fh:
            ffprob_info = json.load(fh)
            ffprob_info_str = json.dumps(ffprob_info, indent=2, ensure_ascii=False)

        return [
            gr.Text.update(label=LABEL_WORKSPACE_FOLDER, value=workspace, interactive=False),
            gr.TextArea.update(value=ffprob_info_str, visible=True),
            gr_update_status(f'open workspace {ws_name!r}'),
        ]

    # try create new workspace
    cmd = f'"{FFPROBE_BIN}" -i "{video_file.name}" -show_streams -of json'
    print(f'>> exec: {cmd}')
    try:
        ffprob_info = json.loads(os.popen(cmd).read().strip())
        ffprob_info_str = json.dumps(ffprob_info, indent=2, ensure_ascii=False)

        ws_dp.mkdir(parents=True)
        workspace = ws_dp
        ws_name = workspace.name

        with open(info_fp, 'w', encoding='utf-8') as fh:
            fh.write(ffprob_info_str)

        return [
            gr.Text.update(label=LABEL_WORKSPACE_FOLDER, value=workspace, interactive=False),
            gr.TextArea.update(value=ffprob_info_str, visible=True),
            gr_update_status(f'create new workspace {ws_name!r}'),
        ]
    except:
        e = format_exc()
        print(e)
        return [
            gr.Text.update(),
            gr.TextArea.update(visible=False),
            gr_update_status(e, code=RetCode.ERROR),
        ]


def _txt_working_folder(working_folder: str) -> GradioRequest:
    global workspace, cur_cache_folder

    # Mode: workspace folder
    if workspace is not None: return gr_update_status()

    # Mode: cache folder
    working_folder: Path = Path(working_folder)
    if working_folder.is_dir():
        cur_cache_folder = working_folder
        return gr_update_status(f'set cache folder path: {cur_cache_folder}')
    else:
        return gr_update_status(f'invalid folder path: {working_folder}', code=RetCode.WARN)


def _btn_open(working_folder: str) -> GradioRequest:
    if Path(working_folder).is_dir():
        os.startfile(working_folder)
        return gr_update_status(f'open folder: {working_folder!r}')
    else:
        return gr_update_status(f'invalid folder path: {working_folder!r}', code=RetCode.ERROR)


def _btn_interrupt() -> GradioRequest:
    global cur_task, cur_proc
    if cur_proc is not None:
        cur_proc.kill()
        cur_proc = None
    state.interrupt()
    cur_task = None
    return gr_update_status('interrupted', code=RetCode.ERROR)


def _chk_allow_overwrite(allow_overwrite: bool) -> None:
    global cur_allow_overwrite

    cur_allow_overwrite = allow_overwrite


def _chk_process_audio(process_audio: bool) -> None:
    global cur_process_audio

    cur_process_audio = process_audio


class Script(scripts.Script):
    def title(self):
        return 'vid2vid'

    def describe(self):
        return 'Convert a video to an AI generated stuff.'

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Row(variant='panel'):
            status_info = gr.HTML()

        with gr.Row(variant='compact').style(equal_height=True):
            working_folder = gr.Text(label=LABEL_CACHE_FOLDER, value=lambda: DEFAULT_CACHE_FOLDER, max_lines=1)
            working_folder.change(fn=_txt_working_folder, inputs=working_folder, outputs=status_info,
                                  show_progress=False)
            btn_open = gr.Button(value='\U0001f4c2', variant='tool')  # 📂
            btn_open.click(fn=_btn_open, inputs=working_folder, outputs=status_info, show_progress=False)

        with gr.Row():
            with gr.Tab('1: Extract frames'):
                with gr.Row(variant='panel'):
                    gr.HTML(EXTRACT_HELP_HTML)

                with gr.Row(variant='compact').style(equal_height=True):
                    video_file = gr.File(label=LABEL_VIDEO_FILE, file_types=['video'])
                    video_info = gr.TextArea(label=LABEL_VIDEO_INFO, max_lines=7, visible=False)
                    video_file.change(fn=_file_select, inputs=video_file,
                                      outputs=[working_folder, video_info, status_info], show_progress=False)

                with gr.Row(variant='compact').style(equal_height=True):
                    extract_fmt = gr.Dropdown(label=LABEL_EXTRACT_FMT, value=lambda: DEFAULT_EXTRACT_FMT,
                                              choices=CHOICES_IMAGE_FMT)
                    extract_frame = gr.Dropdown(label=LABEL_EXTRACT_FRAME, value=lambda: DEFAULT_EXTRACT_FRAME,
                                                choices=CHOICES_EXTRACT_FRAME)
                    extract_fps = gr.Slider(label=LABEL_EXTRACT_FPS, value=lambda: DEFAULT_EXTRACT_FPS, minimum=1,
                                            maximum=24, step=0.1,
                                            visible=ExtractFrame(DEFAULT_EXTRACT_FRAME) == ExtractFrame.FPS)

                    extract_frame.change(fn=lambda x: gr_show(ExtractFrame(x) == ExtractFrame.FPS),
                                         inputs=extract_frame, outputs=extract_fps, show_progress=False)

                    btn_ffmpeg_extract = gr.Button('Extract frames!')
                    btn_ffmpeg_extract.click(fn=_btn_ffmpeg_extract,
                                             inputs=[video_file, extract_frame, extract_fmt, extract_fps],
                                             outputs=status_info, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get ffprobe.json, frames\*.jpg, audio.wav'))

            with gr.Tab('2: Make masks & tags'):
                with gr.Row(variant='panel'):
                    gr.HTML(MATERIAL_HELP_HTML)

                with gr.Row(variant='compact').style(equal_height=True):
                    midas_model = gr.Radio(label=LABEL_MIDAS_MODEL, value=lambda: DEFAULT_MIDAS_MODEL,
                                           choices=CHOICES_MIDAS_MODEL)

                with gr.Row(variant='default').style(equal_height=True):
                    btn_frame_delta = gr.Button('Make frame delta!')
                    btn_frame_delta.click(fn=_btn_frame_delta, outputs=status_info, show_progress=False)

                    btn_midas = gr.Button('Make depth masks!')
                    btn_midas.click(fn=_btn_midas, inputs=midas_model, outputs=status_info, show_progress=False)

                    btn_deepdanbooru = gr.Button('Make inverted tags!')
                    btn_deepdanbooru.click(fn=_btn_deepdanbooru, outputs=status_info, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get framedelta\*.png, depthmask\*.png, tags.json, tags-topk.txt'))

            with gr.Tab('3: Successive img2img'):
                with gr.Row(variant='panel'):
                    gr.HTML(value=IMG2IMG_HELP_HTML)

                with gr.Row(variant='compact'):
                    img2img_mode = gr.Radio(label=LABEL_IMG2IMG_MODE, value=lambda: DEFAULT_IMG2IMG_MODE,
                                            choices=CHOICES_IMG2IMG_MODE)

                with gr.Row(variant='compact').style(equal_height=True):
                    sigma_meth = gr.Dropdown(label=LABEL_SIGMA_METH, value=lambda: DEFAULT_SIGMA_METH,
                                             choices=CHOICES_SIGMA_METH)
                    init_noise_w = gr.Slider(label=LABEL_INIT_NOISE_W, value=lambda: DEFAULT_INIT_NOISE_W, minimum=0.0,
                                             maximum=1.0, step=0.01)

                with gr.Row(visible=SigmaSched(DEFAULT_SIGMA_METH) != SigmaSched.DEFAULT).style(
                        equal_height=True) as tab_sigma_sched:
                    steps = gr.Slider(label=LABEL_STEPS, value=lambda: DEFAULT_STEPS, minimum=1, maximum=150, step=1)
                    denoise_w = gr.Slider(label=LABEL_DENOISE_W, value=lambda: DEFAULT_DENOISE_W, minimum=0.0,
                                          maximum=1.0, step=0.01)
                    sigma_min = gr.Slider(label=LABEL_SIGMA_MIN, value=lambda: DEFAULT_SIGMA_MIN, minimum=0.1,
                                          maximum=5.0, step=0.01)
                    sigma_max = gr.Slider(label=LABEL_SIGMA_MAX, value=lambda: DEFAULT_SIGMA_MAX, minimum=0.1,
                                          maximum=5.0, step=0.01)

                sigma_meth.change(fn=lambda x: gr_show(SigmaSched(x) != SigmaSched.DEFAULT), inputs=sigma_meth,
                                  outputs=tab_sigma_sched, show_progress=False)

                with gr.Group() as tab_extras:
                    with gr.Row(variant='compact').style(equal_height=True):
                        fdc_methd = gr.Dropdown(label=LABEL_FDC_METH, value=lambda: DEFAULT_FDC_METH,
                                                choices=CHOICES_FDC_METH)
                        delta_mask = gr.Dropdown(label=LABEL_DELTA_MASK, value=lambda: DEFAULT_DELTA_MASK,
                                                 choices=CHOICES_MASK)
                        spatial_mask = gr.Dropdown(label=LABEL_SPATIAL_MASK, value=lambda: DEFAULT_SPATIAL_MASK,
                                                   choices=CHOICES_MASK)
                    with gr.Row(variant='compact').style(equal_height=True) as tab_params:
                        motion_highext = gr.Slider(label=LABEL_MOTION_HIGHEXT, value=lambda: DEFAULT_MOTION_HIGHEXT,
                                                   minimum=1, maximum=15, step=2)
                        motion_lowcut = gr.Slider(label=LABEL_MOTION_LOWCUT, value=lambda: DEFAULT_MOTION_LOWCUT,
                                                  minimum=0, maximum=255, step=8)
                        depth_lowcut = gr.Slider(label=LABEL_DEPTH_LOWCUT, value=lambda: DEFAULT_DEPTH_LOWCUT,
                                                 minimum=0, maximum=255, step=8,
                                                 interactive=MaskType(DEFAULT_SPATIAL_MASK) == MaskType.DEPTH)

                    def switch_params(delta_mask, spatial_mask):
                        delta_mask = MaskType(delta_mask)
                        spatial_mask = MaskType(spatial_mask)
                        show_tab = spatial_mask != MaskType.NONE or delta_mask != MaskType.NONE
                        act_motion = spatial_mask == MaskType.MOTION or delta_mask == MaskType.MOTION
                        act_depth = spatial_mask == MaskType.DEPTH or delta_mask == MaskType.DEPTH
                        return [
                            gr_show(show_tab),
                            gr.Slider.update(interactive=act_motion),
                            gr.Slider.update(interactive=act_motion),
                            gr.Slider.update(interactive=act_depth),
                        ]

                    delta_mask.change(fn=switch_params, inputs=[delta_mask, spatial_mask],
                                      outputs=[tab_params, motion_highext, motion_lowcut, depth_lowcut],
                                      show_progress=False)
                    spatial_mask.change(fn=switch_params, inputs=[delta_mask, spatial_mask],
                                        outputs=[tab_params, motion_highext, motion_lowcut, depth_lowcut],
                                        show_progress=False)

                img2img_mode.change(fn=lambda x: gr_show(Img2ImgMode(x) == Img2ImgMode.BATCH), inputs=img2img_mode,
                                    outputs=tab_extras, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get img2img\*.png, motionmask\*.png'))

            with gr.Tab('4: Upscale & interpolate'):
                with gr.Row(variant='panel'):
                    gr.HTML(value=POSTPROCESS_HELP_HTML)

                with gr.Row(variant='compact').style(equal_height=True):
                    resr_model = gr.Dropdown(label=LABEL_RESR_MODEL, value=lambda: DEFAULT_RESR_MODEL,
                                             choices=CHOICES_RESR_MODEL)
                    btn_resr = gr.Button('Launch super-resolution!')
                    btn_resr.click(fn=_btn_resr, inputs=resr_model, outputs=status_info, show_progress=False)

                with gr.Row(variant='compact').style(equal_height=True):
                    rife_model = gr.Dropdown(label=LABEL_RIFE_MODEL, value=lambda: DEFAULT_RIFE_MODEL,
                                             choices=CHOICES_RIFE_MODEL)
                    rife_ratio = gr.Slider(label=LABEL_RIFE_RATIO, value=lambda: DEFAULT_RIFE_RATIO, minimum=0.5,
                                           maximum=4.0, step=0.1)
                    btn_rife = gr.Button('Launch frame-interpolation!')
                    btn_rife.click(fn=_btn_rife, inputs=[rife_model, rife_ratio, extract_fmt], outputs=status_info,
                                   show_progress=False)

                gr.HTML(html.escape(r'=> expected to get resr\*.png, rife\*.png'))

            with gr.Tab('5: Export'):
                with gr.Row(variant='panel'):
                    gr.HTML(value=EXPORT_HELP_HTML)

                with gr.Row(variant='compact').style(equal_height=True):
                    export_fmt = gr.Dropdown(label=LABEL_EXPORT_FMT, value=lambda: DEFAULT_EXPORT_FMT,
                                             choices=CHOICES_VIDEO_FMT)
                    frame_src = gr.Dropdown(label=LABEL_FRAME_SRC, value=lambda: DEFAULT_FRAME_SRC,
                                            choices=CHOICES_FRAME_SRC)

                    btn_ffmpeg_compose = gr.Button('Export!')
                    btn_ffmpeg_compose.click(fn=_btn_ffmpeg_export,
                                             inputs=[export_fmt, frame_src, extract_fmt, extract_fps, extract_frame,
                                                     rife_ratio], outputs=status_info, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get synth-*.mp4'))

        with gr.Row(variant='compact').style(equal_height=True):
            allow_overwrite = gr.Checkbox(label=LABEL_ALLOW_OVERWRITE, value=lambda: DEFAULT_ALLOW_OVERWRITE)
            allow_overwrite.change(fn=_chk_allow_overwrite, inputs=allow_overwrite)

            process_audio = gr.Checkbox(label=LABEL_PROCESS_AUDIO, value=lambda: DEFAULT_PROCESS_AUDIO)
            process_audio.change(fn=_chk_process_audio, inputs=process_audio)

            btn_interrut = gr.Button('Interrupt!', variant='primary')
            btn_interrut.click(fn=_btn_interrupt, outputs=status_info, show_progress=False)

        return [
            img2img_mode,
            init_noise_w, sigma_meth,
            steps, denoise_w, sigma_min, sigma_max,
            fdc_methd, delta_mask, spatial_mask,
            motion_highext, motion_lowcut, depth_lowcut,
        ]

    def run(self, p: StableDiffusionProcessingImg2Img,
            img2img_mode: str,
            init_noise_w: float, sigma_meth: str,
            steps: int, denoise_w: float, sigma_min: float, sigma_max: float,
            fdc_methd: str, delta_mask: str, spatial_mask: str,
            motion_highext: int, motion_lowcut: int, depth_lowcut: int,
            ):

        if sigma_max < sigma_min:
            return Processed(p, [], p.seed, 'error sigma_max < sigma_min!')

        img2img_mode: Img2ImgMode = Img2ImgMode(img2img_mode)
        sigma_meth: SigmaSched = SigmaSched(sigma_meth)
        fdc_methd: FrameDeltaCorrection = FrameDeltaCorrection(fdc_methd)
        spatial_mask: MaskType = MaskType(spatial_mask)
        delta_mask: MaskType = MaskType(delta_mask)

        if img2img_mode == Img2ImgMode.BATCH:
            if workspace is None:
                return Processed(p, [], p.seed, 'no current workspace opened!')

            if 'check cache exists':
                out_dp = workspace / WS_IMG2IMG
                if out_dp.exists():
                    if not cur_allow_overwrite:
                        return Processed(p, [], p.seed, task_ignore_str('img2img'))
                    shutil.rmtree(str(out_dp))
                out_dp.mkdir()

            if 'check required materials exist':
                frames_dp = workspace / WS_FRAMES
                if not frames_dp.exists():
                    return Processed(p, [], p.seed, f'frames folder not found: {frames_dp}')
                n_inits = get_folder_file_count(frames_dp)

                require_delta = any([spatial_mask == MaskType.MOTION, delta_mask == MaskType.MOTION,
                                     fdc_methd != FrameDeltaCorrection.NONE])
                delta_dp = workspace / WS_DFRAME
                if require_delta:
                    if not delta_dp.exists():
                        return Processed(p, [], p.seed, f'framedelta folder not found: {delta_dp}')
                    n_delta = get_folder_file_count(delta_dp)
                    if n_delta != n_inits - 1:
                        return Processed(p, [], p.seed,
                                         f'number mismatch for n_delta ({n_delta}) != n_frames ({n_inits}) - 1')

                require_depth = spatial_mask == MaskType.DEPTH
                depth_dp = workspace / WS_DEPTH
                if require_depth:
                    if not depth_dp.exists():
                        return Processed(p, [], p.seed, f'mask folder not found: {depth_dp}')
                    n_masks = get_folder_file_count(depth_dp)
                    if n_masks != n_inits:
                        return Processed(p, [], p.seed,
                                         f'number mismatch for n_masks ({n_masks}) != n_frames ({n_inits})')

            self.init_dp = frames_dp
            self.delta_dp = delta_dp
            self.depth_dp = depth_dp
            self.fdc_methd = fdc_methd
            self.delta_mask = delta_mask
            self.spatial_mask = spatial_mask
            self.motion_highext = motion_highext
            self.motion_lowcut = motion_lowcut
            self.depth_lowcut = depth_lowcut
        else:
            if workspace is not None:
                out_dp = workspace / WS_IMG2IMG_DEBUG
                out_dp.mkdir(exist_ok=True)
            else:
                out_dp = p.outpath_samples

        if sigma_meth != SigmaSched.DEFAULT:
            from k_diffusion.sampling import get_sigmas_karras, get_sigmas_exponential, get_sigmas_polyexponential, \
                get_sigmas_vp

            sigma_min = max(sigma_min, 1e-3)

            if sigma_meth == SigmaSched.KARRAS:
                sigma_fn = lambda n: get_sigmas_karras(n, sigma_min, sigma_max)
            elif sigma_meth == SigmaSched.EXP:
                sigma_fn = lambda n: get_sigmas_exponential(n, sigma_min, sigma_max)
            elif sigma_meth == SigmaSched.POLY_EXP:
                sigma_fn = lambda n: get_sigmas_polyexponential(n, sigma_min, sigma_max)
            elif sigma_meth == SigmaSched.VP:
                sigma_fn = lambda n: get_sigmas_vp(n, sigma_max, sigma_min)
            elif sigma_meth == SigmaSched.LINEAR:
                sigma_fn = lambda n: torch.linspace(sigma_max, sigma_min, n)

            p.steps = steps
            p.denoising_strength = denoise_w
            p.sampler_noise_scheduler_override = lambda step: sigma_fn(step).to(p.sd_model.device)

        if 'show real sigma':
            real_steps, t_enc = setup_img2img_steps(p)
            sigmas = sigma_fn(steps).numpy().tolist()
            real_sigmas = sigmas[real_steps - t_enc - 1:]
            print(f'>> real sigmas: {real_sigmas}')

        if 'override & fix p settings':
            p.n_iter = 1
            p.batch_size = 1
            p.seed = get_fixed_seed(p.seed)
            p.subseed = get_fixed_seed(p.subseed)
            p.do_not_save_grid = True
            p.do_not_save_samples = False
            p.outpath_samples = str(out_dp)
            p.initial_noise_multiplier = init_noise_w

        def cfg_denoiser_hijack(param: CFGDenoiserParams):
            if not 'show real sigma':
                print(f'>> [{param.sampling_step + 1}/{param.total_sampling_steps}] sigma: {param.sigma[-1].item()}')

        env = globals()
        runner = self.run_batch_img2img if img2img_mode == Img2ImgMode.BATCH else self.run_img2img
        if 'process_images_before' in env and 'process_images_after' in env:
            try:
                on_cfg_denoiser(cfg_denoiser_hijack)
                stored_opts = process_images_before(p)
                self.processer = process_images_inner
                images, info = runner(p)
                process_images_after(p, stored_opts)
            finally:
                remove_callbacks_for_function(cfg_denoiser_hijack)
        else:  # safely fallback when prompt-travel is broken
            try:
                on_cfg_denoiser(cfg_denoiser_hijack)
                self.processer = process_images
                images, info = runner(p)
            finally:
                remove_callbacks_for_function(cfg_denoiser_hijack)

        # show only partial results
        return Processed(p, images[::DEFAULT_EXTRACT_FPS // 4][:100], p.seed, info)

    def run_img2img(self, p: StableDiffusionProcessingImg2Img) -> Tuple[List[PILImage], str]:
        print(f'>> save debug samples to: {p.outpath_samples}')
        proc = self.processer(p)
        return proc.images, proc.info

    def run_batch_img2img(self, p: StableDiffusionProcessingImg2Img) -> Tuple[List[PILImage], str]:
        init_dp = self.init_dp
        delta_dp = self.delta_dp
        depth_dp = self.depth_dp
        fdc_methd = self.fdc_methd
        delta_mask = self.delta_mask
        spatial_mask = self.spatial_mask
        motion_lowcut = self.motion_lowcut
        motion_highext = self.motion_highext
        depth_lowcut = self.depth_lowcut
        init_fns = sorted(os.listdir(init_dp))

        motion_dp = workspace / WS_MOTION
        motion_dp.mkdir(exist_ok=True)

        initial_info: str = None
        images: List[PILImage] = []

        def get_init(idx: int) -> PILImage:
            return get_img(init_dp / init_fns[idx], mode='RGB')

        def get_depth(idx: int, lowcut: int = 0, w: int = None, h: int = None) -> PILImage:
            img = get_img(depth_dp / Path(init_fns[idx]).with_suffix('.png'))
            if all([h, w]): img = resize_image(p.resize_mode, img, w, h)
            return im_to_img(im_mask_lowcut(img_to_im(img), thresh=lowcut / 255.0))  # [0.0, 1.0]

        def get_delta(idx: int, w: int = None, h: int = None) -> npimg:
            img = get_img(delta_dp / Path(init_fns[idx]).with_suffix('.png'))
            if all([h, w]): img = resize_image(p.resize_mode, img, w, h)
            return im_shift_n1p1(img_to_im(img))  # [-1.0, 1.0]

        use_fdc = fdc_methd != FrameDeltaCorrection.NONE
        use_delta_mask = delta_mask != MaskType.NONE
        last_frame: npimg = None
        iframe = 0

        def image_save_hijack(param: ImageSaveParams):
            # just 5-digit serial number, starts from '00001'
            fp = Path(param.filename)
            sn = int(fp.stem[:5])
            if sn <= 0: sn += 1
            fn = f'{sn:05d}' + fp.suffix
            param.filename = str(fp.parent / fn)

            # force RGB mode, RIFE not work on RGBA
            param.image = param.image.convert('RGB')

            if not any([use_fdc, use_delta_mask]): return

            nonlocal last_frame
            if last_frame is not None:
                this_frame = img_to_im(param.image)  # [0.0, 1.0]
                H, W, C = this_frame.shape
                tgt_d = get_delta(iframe, W, H)  # [-1.0, 1.0]

                if use_delta_mask:
                    if delta_mask == MaskType.MOTION:
                        mask = im_delta_to_motion(tgt_d, motion_lowcut / 255.0, expand=motion_highext)  # [0.0, 1.0]
                        im_to_img(mask).save(motion_dp / Path(init_fns[iframe]).with_suffix('.png'))  # for debug

                        cur_d = this_frame - last_frame  # [-1.0, 1.0]
                        this_frame = last_frame + cur_d * mask

                    elif delta_mask == MaskType.DEPTH:
                        mask = get_depth(iframe, depth_lowcut, W, H)

                        this_frame = this_frame * mask + last_frame * (1 - mask)

                    if not 'debug':
                        dd = np.abs(this_frame - last_frame)
                        print(f'>> motion correction max: {dd.max()}, mean: {dd.mean()}')

                if use_fdc:
                    cur_d = this_frame - last_frame  # [-1.0, 1.0]

                    if fdc_methd == FrameDeltaCorrection.CLIP:
                        new_d = cur_d.clip(tgt_d.min(), tgt_d.max())
                    else:
                        cur_d_n, (cur_avg, cur_std) = im_norm(cur_d, ret_stats=True)
                        tgt_d_n, (tgt_avg, tgt_std) = im_norm(tgt_d, ret_stats=True)

                        if fdc_methd == FrameDeltaCorrection.AVG:
                            new_d = cur_d_n * cur_std + tgt_avg
                        elif fdc_methd == FrameDeltaCorrection.STD:
                            new_d = cur_d_n * tgt_std + cur_avg
                        elif fdc_methd == FrameDeltaCorrection.NORM:
                            new_d = cur_d_n * tgt_std + tgt_avg

                    this_frame = last_frame + new_d

                    if not 'debug':
                        dd = np.abs(this_frame - last_frame)
                        print(f'>> stats correction max: {dd.max()}, mean: {dd.mean()}')

                this_frame = im_clip(this_frame)
                param.image = im_to_img(this_frame)

                last_frame = this_frame
            else:
                last_frame = img_to_im(param.image)

        try:
            on_before_image_saved(image_save_hijack)

            n_frames = len(init_fns)
            state.job_count = n_frames
            for i in tqdm(range(n_frames)):
                if state.interrupted: break

                state.job = f'{i}/{n_frames}'
                state.job_no = i + 1
                iframe = i

                p.init_images = [get_init(i)]

                if spatial_mask == MaskType.DEPTH:
                    p.image_mask = get_depth(i, depth_lowcut)
                elif spatial_mask == MaskType.MOTION and i > 0:  # ignore the first frame
                    delta = get_delta(i, p.width, p.height)
                    mask = im_delta_to_motion(delta, motion_lowcut / 255.0, expand=motion_highext)
                    p.image_mask = im_to_img(mask)

                proc = self.processer(p)
                if initial_info is None: initial_info = proc.info
                images.extend(proc.images)
        finally:
            remove_callbacks_for_function(image_save_hijack)

        return images, initial_info
