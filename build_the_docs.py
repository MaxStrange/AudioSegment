"""
This script builds the documentation for this project. Just call it with python3 and it
should just build the docs, regardless of your current platform. It will prompt you to
install the necessary requirements.
"""
import audiosegment
import os
import shutil
import subprocess
import sys

if shutil.which("sphinx-apidoc") is None:
    print("You must install sphinx and make sure that sphinx-apidoc is in your path.")
    exit(1)
try:
    import sphinx_rtd_theme
except ImportError:
    print("You must install sphinx_rtd_theme. Try using: `pip3 install sphinx_rtd_theme`")
    exit(1)

COPYRIGHT = "Max Strange (MIT License)"
AUTHOR = "Max Strange"
PROJECT = "AudioSegment"

python_interp = "python3" if shutil.which("python3") is not None else "python"
home = os.path.dirname(os.path.realpath(__file__))
apipath = os.sep.join([home, "docs", "api"])
project_setup_py_path = os.sep.join([home, "setup.py"])

if __name__ == "__main__":
    # remove any old docs if there were any
    shutil.rmtree(apipath, ignore_errors=True)
    shutil.rmtree(os.sep.join([home, "__pycache__"]), ignore_errors=True)
    try:
        os.remove(os.sep.join([home, "conf.py"]))
    except FileNotFoundError:
        pass
    # Remake the directory and execute sphinx-apidoc on it
    try:
        os.makedirs(apipath)
    except FileExistsError:
        print("Is the API directory open in another program? Couldn't remove it completely. May not matter.")
    result = subprocess.run(["sphinx-apidoc", "--force", "-o", apipath, "--full", "-H", PROJECT, home, "exclude_pattern", "setup.py", "setup.cfg", "build_the_docs.py", "conf.py"])
    result.check_returncode()
    shutil.copyfile(os.sep.join([home, "audiosegment.py"]), os.sep.join([apipath, "audiosegment.py"]))

    # get the version from setup.py
    version = None
    with open(project_setup_py_path) as f:
        for line in f:
            if line.strip().startswith("version=\""):
                version = line.strip().split("version=")[1].rstrip(",").strip("\"")
                break
    assert version is not None

    # Now write the conf.py file
    with open(os.sep.join([apipath, "conf.py"])) as f:
        lines = [line for line in f]
    for i, line in enumerate(lines):
        if line.startswith("project = "):
            lines[i] = "project = '" + PROJECT + "'" + os.linesep
        elif line.startswith("copyright = "):
            lines[i] = "copyright = '" + COPYRIGHT + "'" + os.linesep
        elif line.startswith("author = "):
            lines[i] = "author = '" + AUTHOR + "'" + os.linesep
        elif line.startswith("version = "):
            lines[i] = "version = '" + version + "'" + os.linesep
        elif line.startswith("html_theme = "):
            lines[i] = "html_theme = 'sphinx_rtd_theme'" + os.linesep
        elif line.startswith("pygments_style = "):
            lines[i] = "pygments_style = 'autumn'" + os.linesep
    with open(os.sep.join([apipath, "conf.py"]), 'w') as f:
        for line in lines:
            f.write(line)

    # Now change the interpreter in the Makefile
    with open(os.sep.join([apipath, "Makefile"])) as f:
        lines = [line for line in f]
    for i, line in enumerate(lines):
        if line.startswith("SPHINXBUILD   ="):
            lines[i] = "SPHINXBUILD   = " + python_interp + " -m sphinx" + os.linesep
    with open(os.sep.join([apipath, "Makefile"]), 'w') as f:
        for line in lines:
            f.write(line)

    # Change the interpreter in make.bat
    with open(os.sep.join([apipath, "make.bat"])) as f:
        lines = [line for line in f]
    for i, line in enumerate(lines):
        if line.startswith("\tset SPHINXBUILD="):
            lines[i] = "\tset SPHINXBUILD=" + python_interp + " -m sphinx" + os.linesep
    with open(os.sep.join([apipath, "make.bat"]), 'w') as f:
        for line in lines:
            f.write(line)

    # Copy images into the right directory so they can show up in the docs
    image_folder_path = os.sep.join([apipath, "..", "images"])
    fft_path = os.sep.join([image_folder_path, "fft.png"])
    spectrogram_path = os.sep.join([image_folder_path, "spectrogram.png"])
    filter_path = os.sep.join([image_folder_path, "filter_bank.png"])
    new_image_folder_path = os.sep.join([apipath, "images"])
    os.makedirs(new_image_folder_path)
    shutil.copyfile(fft_path, os.sep.join([new_image_folder_path, "fft.png"]))
    shutil.copyfile(spectrogram_path, os.sep.join([new_image_folder_path, "spectrogram.png"]))
    shutil.copyfile(filter_path, os.sep.join([new_image_folder_path, "filter_bank.png"]))

    # cd into the Makefile's directory and execute make clean and make
    os.chdir(apipath)
    if sys.platform == "win32":
        if not "SPHINXBUILD" in os.environ:
            os.environ["SPHINXBUILD"] = python_interp + " -m sphinx"
        result = subprocess.run(["make.bat", "clean"])
        result.check_returncode()
        result = subprocess.run(["make.bat", "html"])
    else:
        result = subprocess.run(["make", "clean"])
        result = subprocess.run(["make", "html"])
    result.check_returncode()

