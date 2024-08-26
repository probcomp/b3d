#!/usr/bin/env python

import os
import subprocess


def launch_python_module():
    command = ["python", "-m", "rerun", "--port", "8812"]
    with open(os.devnull, "w") as DEVNULL:
        subprocess.Popen(command, stdout=DEVNULL, stderr=DEVNULL)


if __name__ == "__main__":
    launch_python_module()
