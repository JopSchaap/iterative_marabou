import os
from pathlib import Path
import subprocess as sub_proc

# CONSTANTS -------------------------------------------------------------------
MARABOU_DIR = Path("marabou")
BUILD_DIR = MARABOU_DIR / Path("build")
MARABOUD_GIT_URL = "https://github.com/NeuralNetworkVerification/Marabou.git"
PROC_NUM = None # Use none for system maximum
# -----------------------------------------------------------------------------
PROC_NUM = PROC_NUM or len(os.sched_getaffinity(0))


def setup_mabou(build_dir = BUILD_DIR, marabou_dir = MARABOU_DIR):
    sub_proc.run(["git", "clone", MARABOUD_GIT_URL, marabou_dir])
    
    build_dir.mkdir(parents=True, exist_ok=True)

    sub_proc.run(["cmake", marabou_dir.absolute(), "-DBUILD_PYTHON=ON"], cwd=BUILD_DIR, check=True)
    sub_proc.run(["cmake", "-S", marabou_dir.absolute(),"--build", ".", "--parallel", PROC_NUM], cwd=BUILD_DIR, check=True)



if __name__ == "__main__":
    setup_mabou()