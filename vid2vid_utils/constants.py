from pathlib import Path
from modules.shared import opts
from .enums import *
from subprocess import Popen

from vid2vid_utils.external_tools import get_resr_model_names, get_rife_model_names, SD_WEBUI_PATH

OUTPUT_PATH = SD_WEBUI_PATH / 'outputs'


def __(key, value=None):
    k1 = f'customscript/vid2vid.py/img2img/{key}/value'
    if k1 in opts.data: return opts.data.get(k1, value)
    k2 = f'img2img/{key}/value'
    return opts.data.get(k2, value)


WS_FFPROBE = 'ffprobe.json'
WS_FRAMES = 'frames'
WS_AUDIO = 'audio.wav'
WS_DFRAME = 'framedelta'
WS_MOTION = 'motionmask'
WS_DEPTH = 'depthmask'  # only for debug, not for prepare
WS_TAGS = 'tags.json'
WS_TAGS_TOPK = 'tags-topk.txt'
WS_IMG2IMG = 'img2img'
WS_IMG2IMG_DEBUG = 'img2img.debug'
WS_RESR = 'resr'
WS_RIFE = 'rife'
WS_SYNTH = 'synth'  # stem


LABEL_CACHE_FOLDER = 'Cache Folder'
LABEL_WORKSPACE_FOLDER = 'Workspace Folder'
LABEL_VIDEO_FILE = 'Input video file'
LABEL_VIDEO_INFO = 'Video media info'
LABEL_EXTRACT_FRAME = 'Extract frames'
LABEL_EXTRACT_FMT = 'Extract format'
LABEL_EXTRACT_FPS = 'Extract FPS'
LABEL_MIDAS_MODEL = 'MiDaS model (depthmap)'
LABEL_IMG2IMG_MODE = 'Img2Img mode'
LABEL_SIGMA_METH = 'Override sigma schedule'
LABEL_STEPS = 'Sampling steps'
LABEL_DENOISE_W = 'Denoising strength'
LABEL_SIGMA_MIN = 'Sigma min'
LABEL_SIGMA_MAX = 'Sigma max'
LABEL_INIT_NOISE_W = 'Init noise weight'
LABEL_FDC_METH = 'Statistical correction'
LABEL_SPATIAL_MASK = 'Spatial mask'
LABEL_DELTA_MASK = 'Delta mask'
LABEL_MOTION_HIGHEXT = 'Motion high-ext'
LABEL_MOTION_LOWCUT = 'Motion low-cut'
LABEL_DEPTH_LOWCUT = 'Depth low-cut'
LABEL_RESR_MODEL = 'Real-ESRGAN model (image upscale)'
LABEL_RIFE_MODEL = 'RIFE model (video interp)'
LABEL_RIFE_RATIO = 'Interpolation ratio'
LABEL_EXPORT_FMT = 'Export format'
LABEL_FRAME_SRC = 'Frame source'
LABEL_ALLOW_OVERWRITE = 'Allow overwrite cache'
LABEL_PROCESS_AUDIO = 'Process audio'

CHOICES_EXTRACT_FRAME = [x.value for x in ExtractFrame]
CHOICES_IMAGE_FMT = [x.value for x in ImageFormat]
CHOICES_VIDEO_FMT = [x.value for x in VideoFormat]
CHOICES_SIGMA_METH = [x.value for x in SigmaSched]
CHOICES_FDC_METH = [x.value for x in FrameDeltaCorrection]
CHOICES_MIDAS_MODEL = [x.value for x in MidasModel]
CHOICES_IMG2IMG_MODE = [x.value for x in Img2ImgMode]
CHOICES_MASK = [x.value for x in MaskType]
CHOICES_RESR_MODEL = get_resr_model_names()
CHOICES_RIFE_MODEL = get_rife_model_names()
CHOICES_FRAME_SRC = [
    WS_FRAMES,
    WS_DFRAME,
    WS_MOTION,
    WS_DEPTH,
    WS_IMG2IMG,
    WS_RESR,
    WS_RIFE,
]

INIT_CACHE_FOLDER = OUTPUT_PATH / 'sd-webui-vid2vid'
INIT_CACHE_FOLDER.mkdir(exist_ok=True)

DEFAULT_CACHE_FOLDER = __(LABEL_CACHE_FOLDER, str(INIT_CACHE_FOLDER))
DEFAULT_EXTRACT_FRAME = __(LABEL_EXTRACT_FRAME, ExtractFrame.FPS.value)
DEFAULT_EXTRACT_FMT = __(LABEL_EXTRACT_FMT, ImageFormat.PNG.value)
DEFAULT_EXTRACT_FPS = __(LABEL_EXTRACT_FPS, 12)
DEFAULT_MIDAS_MODEL = __(LABEL_MIDAS_MODEL, MidasModel.DPT_LARGE.value)
DEFAULT_IMG2IMG_MODE = __(LABEL_IMG2IMG_MODE, Img2ImgMode.BATCH.value)
DEFAULT_STEPS = __(LABEL_STEPS, 20)
DEFAULT_DENOISE_W = __(LABEL_DENOISE_W, 0.75)
DEFAULT_INIT_NOISE_W = __(LABEL_INIT_NOISE_W, 1.0)
DEFAULT_SIGMA_METH = __(LABEL_SIGMA_METH, SigmaSched.EXP.value)
DEFAULT_SIGMA_MIN = __(LABEL_SIGMA_MIN, 0.1)
DEFAULT_SIGMA_MAX = __(LABEL_SIGMA_MAX, 1.2)
DEFAULT_FDC_METH = __(LABEL_FDC_METH, FrameDeltaCorrection.STD.value)
DEFAULT_DELTA_MASK = __(LABEL_DELTA_MASK, MaskType.MOTION.value)
DEFAULT_SPATIAL_MASK = __(LABEL_SPATIAL_MASK, MaskType.NONE.value)
DEFAULT_MOTION_HIGHEXT = __(LABEL_MOTION_HIGHEXT, 9)
DEFAULT_MOTION_LOWCUT = __(LABEL_MOTION_LOWCUT, 127)
DEFAULT_DEPTH_LOWCUT = __(LABEL_DEPTH_LOWCUT, -1)
DEFAULT_RESR_MODEL = __(LABEL_RESR_MODEL, 'realesr-animevideov3-x2')
DEFAULT_RIFE_MODEL = __(LABEL_RIFE_MODEL, 'rife-v4')
DEFAULT_RIFE_RATIO = __(LABEL_RIFE_RATIO, 2.0)
DEFAULT_FRAME_SRC = __(LABEL_FRAME_SRC, WS_RIFE)
DEFAULT_EXPORT_FMT = __(LABEL_EXPORT_FMT, VideoFormat.MP4.value)
DEFAULT_ALLOW_OVERWRITE = __(LABEL_ALLOW_OVERWRITE, True)
DEFAULT_PROCESS_AUDIO = __(LABEL_PROCESS_AUDIO, False)

EXTRACT_HELP_HTML = '''
    <div>
      <h4> Create a workspace to start everything: 🤗 </h4>
      <p> 1. enter a path for <strong>Cache Folder</strong> to store all things </p>
      <p> 2. open a video file, this will auto-create or reuse a <strong>Worspace Folder</strong> </p>
      <p> 3. check "Process audio" if you want to keep the audio track </p>
      <p> 4. just extract the frames </p>
    </div>
'''

MATERIAL_HELP_HTML = '''
    <div>
      <h4> Preparation for extra materials (Optional): 😧 </h4>
      <p> 1. frame deltas are for statistical correction & motion mask </p>
      <p> 2. depth masks are just literally depth mask... </p>
      <p> 3. inverted tags are for your reference to <strong>manually</strong> fill the prompt box </p>
    </div>
'''

IMG2IMG_HELP_HTML = '''
    <div>
      <h4> Make the conversion magic: 😉 </h4>
      <p> 1. check settings below, and also <strong>all img2img settings</strong> top above ↑↑: prompts, sampler, size, seed, etc.. </p>
      <p> 2. remeber to add <strong>extra networks</strong> or <strong>embeddings</strong> as you want </p>
      <p> 3. <strong>select a dummy init image</strong> in the img2img ref-image box to avoid webui's "AttributeError" error :) </p>
      <p> 4. click the top-right master <strong>"Generate"</strong> button to go! </p>
    </div>
'''

POSTPROCESS_HELP_HTML = '''
    <div>
      <h4> Post-processing for quality and smoothness (Optional): 😆 </h4>
      <p> 1. Real-ESRGAN for image super-resolution, number x2/x3/x4 are the upscale ratio </p>
      <p> 2. RIFE for video frame-interpolation </p>
      <p> 3. data flow for this pipeline is fixed: <strong> Successive img2img -> Real-ESRGAN -> RIFE </strong> </p>
    </div>
'''

EXPORT_HELP_HTML = '''
    <div>
      <h4> Export final results: 😆 </h4>
      <p> 1. usually your wanted frame source is one of "img2img", "resr" or "rife" </p>
      <p> 2. final video's real fps will be auto-calc to match the original speed, no worry~ </p>
      <p> 3. "Process audio" will not work if it was not checked in Step 1 </p>
    </div>
'''


# TODO: fix global variables