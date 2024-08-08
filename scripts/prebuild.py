import shutil

fps_cu_to_cpp = ["src/LISAResponse"]
fps_pyx = ["src/responselisa"]

for fp in fps_cu_to_cpp:
    shutil.copy(fp + ".cu", fp + ".cpp")

for fp in fps_pyx:
    shutil.copy(fp + ".pyx", fp + "_cpu.pyx")


# setup version file
with open("README.md", "r") as fh:
    lines = fh.readlines()

for line in lines:
    if line.startswith("Current Version"):
        version_string = line.split("Current Version: ")[1].split("\n")[0]

with open("fastlisaresponse/_version.py", "w") as f:
    f.write("__version__ = '{}'".format(version_string))

import requests
