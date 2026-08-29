"""Cross-platform subprocess flags for background platform tasks."""

from __future__ import annotations

import os
import subprocess


def hidden_creationflags(*, new_process_group: bool = False, detached: bool = False) -> int:
    """Return flags that keep Windows background commands console-free."""

    if os.name != "nt":
        return 0
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if new_process_group:
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if detached:
        flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    return flags
