import os
import sys
from pathlib import Path
from traceback import print_exc
from typing import List

SD_WEBUI_PATH = Path.cwd()  # should be <sd-webui> root abspath

RESR_BIN = Path('realesrgan-ncnn-vulkan')
RESR_PATH = Path('realesrgan-ncnn-vulkan')
RIFE_BIN = Path('rife-ncnn-vulkan')
RIFE_PATH = Path('rife-ncnn-vulkan')
FFMPEG_BIN = Path('ffmpeg')
FFPROBE_BIN = Path('ffprobe')

PTRAVEL_PATH = SD_WEBUI_PATH / 'extensions' / 'stable-diffusion-webui-prompt-travel'
TOOL_PATH = PTRAVEL_PATH / 'tools'

MIDAS_REPO_PATH = SD_WEBUI_PATH / 'repositories' / 'midas'
MIDAS_MODEL_PATH = SD_WEBUI_PATH / 'models' / 'midas'

if os.name == "posix":
    CURL_BIN = 'curl'

    for d in os.getenv('PATH').split(':'):
        d_path = Path(d)

        if d_path.exists():
            for f in d_path.iterdir():
                if f.is_file():
                    if RESR_BIN.name == f.name:
                        RESR_BIN = f
                    if RIFE_BIN.name == f.name:
                        RIFE_BIN = f
                    if FFMPEG_BIN.name == f.name:
                        FFMPEG_BIN = f
                    if FFPROBE_BIN.name == f.name:
                        FFPROBE_BIN = f

    for d in os.getenv('XDG_DATA_DIRS').split(':'):
        d_path = Path(d)

        if d_path.exists():
            for f in d_path.iterdir():
                if f.is_dir():
                    if RESR_PATH.name == f.name:
                        RESR_PATH = f
                    if RIFE_PATH.name == f.name:
                        RIFE_PATH = f
else:
    # general tool
    CURL_BIN = 'curl.exe'
    # bundled tools
    RESR_PATH = TOOL_PATH / 'realesrgan-ncnn-vulkan'
    RESR_BIN = RESR_PATH / f'{RESR_BIN.name}.exe'

    RIFE_PATH = TOOL_PATH / 'rife-ncnn-vulkan'
    RIFE_BIN = RIFE_PATH / f'{RIFE_BIN.name}.exe'

    FFMPEG_BIN = TOOL_PATH / 'ffmpeg' / 'bin' / f'{FFMPEG_BIN.name}.exe'
    FFPROBE_BIN = TOOL_PATH / 'ffmpeg' / 'bin' / f'{FFPROBE_BIN.name}.exe'

try:
    assert RESR_BIN.exists()
    assert RIFE_BIN.exists()
    assert FFPROBE_BIN.exists()
    assert FFMPEG_BIN.exists()

    assert PTRAVEL_PATH.exists()
    assert TOOL_PATH.exists()
    sys.path.insert(0, str(PTRAVEL_PATH))

    assert MIDAS_REPO_PATH.exists()
except AssertionError:
    print_exc()
    raise RuntimeError('<< integrity check failed, please check your installation :(')


def get_resr_model_names() -> List[str]:
    return sorted({fn.stem for fn in (RESR_PATH / 'models').iterdir()})


def get_rife_model_names() -> List[str]:
    #return [fn.name for fn in RIFE_PATH.iterdir() if fn.is_dir()]
    return ['rife-v4']  # TODO: `only rife-v4 model support custom numframe and timestep`
