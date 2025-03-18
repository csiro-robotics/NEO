import subprocess

from nerfstudio.utils.rich_utils import CONSOLE


def run_subprocess(command):
    """Helper function to run shell commands with real-time output."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    for line in process.stdout:
        CONSOLE.print(line, end="")
    for line in process.stderr:
        CONSOLE.print(line, end="", style="bold red")
    process.stdout.close()
    process.stderr.close()
    process.wait()