import os
from pathlib import Path
import re
import shutil
import subprocess as sub_proc
import sys
import time

# CONSTANTS -------------------------------------------------------------------
MARABOU_DIR = Path("marabou")
BUILD_DIR = MARABOU_DIR / Path("build")
MARABOU_SETUP_PY = MARABOU_DIR / "setup.py"
MARABOU_GIT_URL = "https://github.com/NeuralNetworkVerification/Marabou.git"
PROC_NUM = None # Use none for system maximum
JOB_REGEX = re.compile("\"-j[0-9]+\"")
REPLACE_DOWNLOAD_PROTOBUF_FILE = Path("download_protobuf.sh")
ORIGINAL_DOWNLOAD_PROTOBUF_FILE = MARABOU_DIR / "tools/download_protobuf.sh"
# -----------------------------------------------------------------------------
PROC_NUM = PROC_NUM or len(os.sched_getaffinity(0))


def replace_parallel():
    try:
        txt = MARABOU_SETUP_PY.read_text()
        new_txt = JOB_REGEX.sub(f"\"-j{PROC_NUM}\"", txt)
        MARABOU_SETUP_PY.write_text(new_txt)
    except Exception as e:
        print(e)

def replace_protobuf():
    os.remove(ORIGINAL_DOWNLOAD_PROTOBUF_FILE)
    shutil.copy(REPLACE_DOWNLOAD_PROTOBUF_FILE, ORIGINAL_DOWNLOAD_PROTOBUF_FILE)
 
def setup_marabou(build_dir = BUILD_DIR, marabou_dir = MARABOU_DIR):
    if not marabou_dir.exists():
        sub_proc.run(["git", "clone", "--branch", "v1.0.0", MARABOU_GIT_URL, marabou_dir])
    
    build_dir.mkdir(parents=True, exist_ok=True)

    replace_parallel()
    replace_protobuf()
    sub_proc.run(["bash", ORIGINAL_DOWNLOAD_PROTOBUF_FILE.absolute()] ,cwd=ORIGINAL_DOWNLOAD_PROTOBUF_FILE.parent, check=True)
    sub_proc.run([sys.executable, "setup.py", "install"], cwd=MARABOU_DIR.absolute() ,check=True)

 

if __name__ == "__main__":
    setup_marabou()