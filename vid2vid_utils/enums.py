from enum import Enum
from typing import Dict, Any


class ImageFormat(Enum):
    PNG = 'png'
    JPG = 'jpg'
    WEBP = 'webp'


class VideoFormat(Enum):
    MP4 = 'mp4'
    GIF = 'gif'
    WEBM = 'webm'
    AVI = 'avi'


class ExtractFrame(Enum):
    # ref: https://ottverse.com/i-p-b-frames-idr-keyframes-differences-usecases/
    FPS = '(fixed FPS)'
    IPB = 'I/P/B frames all'
    I = 'I frames only'
    P = 'P frames only'
    B = 'B frames only'


class MidasModel(Enum):
    DPT_LARGE = 'dpt_large'
    DPT_HYBRID = 'dpt_hybrid'
    MIDAS_V21 = 'midas_v21'
    MIDAS_V21_SMALL = 'midas_v21_small'


class Img2ImgMode(Enum):
    BATCH = 'batch img2img'
    SINGLE = 'single img2img (for debug)'


class SigmaSched(Enum):
    DEFAULT = '(use default)'
    KARRAS = 'karras'
    EXP = 'exponential'
    POLY_EXP = 'poly-exponential'
    VP = 'vp'
    LINEAR = 'linear'


class FrameDeltaCorrection(Enum):
    NONE = '(none)'
    CLIP = 'clip min & max'
    AVG = 'shift mean'
    STD = 'shift std'
    NORM = 'shift mean & std'


class MaskType(Enum):
    NONE = '(none)'
    MOTION = 'motion'
    DEPTH = 'depth'


class RetCode(Enum):
    INFO  = 'info'
    WARN  = 'warn'
    ERROR = 'error'


GradioRequest = Dict[str, Any]
