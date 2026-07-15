"""Checked subprocess execution without invoking a shell."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from nerfstudio.utils.rich_utils import CONSOLE


CommandArg = str | Path | int | float


def run_command(command: Sequence[CommandArg], *, verbose: bool = True, dry_run: bool = False) -> None:
    """Run a command as an argument vector and fail on a non-zero exit code."""
    args = [str(arg) for arg in command]
    if not args:
        raise ValueError("Command cannot be empty")
    if verbose or dry_run:
        CONSOLE.print(f"[dim]$ {shlex.join(args)}[/dim]")
    if not dry_run:
        subprocess.run(args, check=True)
