#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PYTHON="$SCRIPT_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "项目环境不存在，面板未启动。"
    exit 0
fi

exec "$VENV_PYTHON" "$SCRIPT_ROOT/panel_service.py" stop
