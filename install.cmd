@REM Auto download and setup repo & tools tools
@ECHO OFF
SETLOCAL

TITLE Setup repo and tools vid2vid...
CD %~dp0

REM paths to web resources
SET SD_PATH=%~dp0..\..
IF NOT EXIST "%SD_PATH%" GOTO die
PUSHD %SD_PATH%
SET SD_PATH=%CD%
POPD

SET EXT_PATH=%SD_PATH%\extensions
IF NOT EXIST "%EXT_PATH%" GOTO die
SET REPO_PATH=%SD_PATH%\repositories
IF NOT EXIST "%REPO_PATH%" GOTO die
SET MODEL_PATH=%SD_PATH%\models
IF NOT EXIST "%MODEL_PATH%" GOTO die

SET GIT_BIN=git.exe
SET CURL_BIN=curl.exe -L -C -

SET PTRAVEL_URL=https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel.git
SET PTRAVEL_PATH=%EXT_PATH%\stable-diffusion-webui-prompt-travel

REM keep compatible with `extensions/depthmap2mask`
SET MIDAS_URL=https://github.com/isl-org/MiDaS.git
SET MIDAS_REPO_PATH=%REPO_PATH%\midas
SET MIDAS_MODEL_PATH=%MODEL_PATH%\midas
REM "dpt_large"
SET MIDAS_MODEL_URL1=https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
REM "midas_v21"
SET MIDAS_MODEL_URL2=https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt
REM "midas_v21_small"
SET MIDAS_MODEL_URL3=https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt
SET MIDAS_MODEL_URL=%MIDAS_MODEL_URL3%
SET MIDAS_MODEL_FILE=%MIDAS_MODEL_PATH%\midas_v21_small-70d6b9c8.pt

REM SET DDB_URL=https://github.com/AUTOMATIC1111/TorchDeepDanbooru
REM SET DDB_MODEL_DIR=deepdanbooru

REM show info for debug
ECHO SD_PATH          = %SD_PATH%
ECHO EXT_PATH         = %EXT_PATH%
ECHO REPO_PATH        = %REPO_PATH%
ECHO MODEL_PATH       = %MODEL_PATH%
ECHO PTRAVEL_PATH     = %PTRAVEL_PATH%
ECHO MIDAS_REPO_PATH  = %MIDAS_REPO_PATH%
ECHO MIDAS_MODEL_PATH = %MIDAS_MODEL_PATH%
ECHO MIDAS_MODEL_FILE = %MIDAS_MODEL_FILE%
ECHO.

REM setup prompt-travel
ECHO ==================================================

ECHO [1/3] clone prompt-travel
IF EXIST %PTRAVEL_PATH% GOTO skip_ptravel
%GIT_BIN% clone %PTRAVEL_URL% %PTRAVEL_PATH%
IF ERRORLEVEL 1 GOTO die
:skip_ptravel

IF EXIST tools GOTO skip_ptravel_tools
MKLINK /J tools %PTRAVEL_PATH%\tools
PUSHD tools
ECHO ^>^> start a new process for installing tools :^)
START cmd.exe /C install.cmd
IF ERRORLEVEL 1 GOTO die
POPD
:skip_ptravel_tools

REM install extra tools
ECHO ==================================================

ECHO [2/3] install MiDaS
IF EXIST %MIDAS_REPO_PATH% GOTO skip_midas
%GIT_BIN% clone %MIDAS_URL% %MIDAS_REPO_PATH%
IF ERRORLEVEL 1 GOTO die
:skip_midas

IF EXIST %MIDAS_MODEL_FILE% GOTO skip_midas_model
%CURL_BIN% %MIDAS_MODEL_URL% -o %MIDAS_MODEL_FILE%
IF ERRORLEVEL 1 GOTO die
:skip_midas_model

ECHO ==================================================

ECHO [3/3] install DeepDanBooru (use built-in)
REM currently do nothing

ECHO ==================================================

REM finished
ECHO ^>^> Done!
ECHO.
GOTO :end

REM error handle
:die
ECHO ^<^< Error!
ECHO ^<^< errorlevel: %ERRORLEVEL%

:end
PAUSE
