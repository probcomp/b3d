#!/usr/bin/env python

import os
import subprocess


def launch_python_module():
    command = ["python", "-m", "rerun", "--port", "8812"]
    with open(os.devnull, "w") as DEVNULL:
        process = subprocess.Popen(
            command, stdout=DEVNULL, stderr=DEVNULL, stdin=subprocess.PIPE
        )
        process.communicate(input=b"yes\n" * 10)


if __name__ == "__main__":
    launch_python_module()
