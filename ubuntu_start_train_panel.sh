#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PYTHON="$SCRIPT_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "尚未安装项目环境，先运行：bash ubuntu_install_and_start.sh" >&2
    exit 1
fi

exec "$VENV_PYTHON" "$SCRIPT_ROOT/panel_service.py" start --no-browser
