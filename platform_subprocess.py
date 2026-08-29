"""Cross-platform subprocess flags for background platform tasks."""

from __future__ import annotations

import os
import subprocess


def hidden_creationflags(*, new_process_group: bool = False) -> int:
    """Return flags that keep Windows background commands console-free.

    Do not combine ``CREATE_NO_WINDOW`` with ``DETACHED_PROCESS``.  On Windows
    11 systems that delegate consoles to Windows Terminal, the detached flag
    can cause a visible Terminal window to be created for the child process.
    Windows processes continue running after their parent exits without it.
    """

    if os.name != "nt":
        return 0
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if new_process_group:
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return flags
