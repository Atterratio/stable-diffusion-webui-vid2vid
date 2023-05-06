import os
import html
import gradio as gr
from pathlib import Path
from subprocess import Popen
from typing import Union
from vid2vid_utils.enums import *


def get_folder_file_count(dp:Union[Path, str]) -> int:
    return len(os.listdir(dp))

test = Path()

def get_file_size(fp:Union[Path, str]) -> float:
    return os.path.getsize(fp) / 2**20


def get_workspace_name(fn:str) -> str:
    name = fn.replace(' ', '_')
    name = name[:32]        # just make things short
    return Path(name).stem


def sh(cmd:str) -> None:
    global cur_proc
    print(f'>> exec: {cmd}')
    cur_proc = Popen(cmd, shell=True, text=True, encoding='utf-8')
    cur_proc.wait()


def gr_update_status(text=None, code=RetCode.INFO, task: str = None, ts: float = None) -> GradioRequest:
    if not text: return gr.HTML.update()
    safe_text = html.escape(text)
    task_str = f' {task!r}' if task else ''
    ts_str = f' ({ts:.3f}s)' if ts else ''
    TEMPLATES = {
        RetCode.INFO: lambda: gr.HTML.update(
            value=f'<div style="padding:10px; color:blue">Done{task_str}!{ts_str} => {safe_text}</div>'),
        RetCode.WARN: lambda: gr.HTML.update(
            value=f'<div style="padding:10px; color:green">Warn{task_str}! => {safe_text}</div>'),
        RetCode.ERROR: lambda: gr.HTML.update(
            value=f'<div style="padding:10px; color:red">Error{task_str}! => {safe_text}</div>'),
    }
    return TEMPLATES[code]()


def task_ignore_str(taskname: str) -> str:
    return f'task "{taskname}" ignored due to cached already exists :)'
