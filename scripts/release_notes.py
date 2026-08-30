"""Extract one version section from CHANGELOG.md for a GitHub Release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADING = re.compile(r"^##\s+v?(?P<version>\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\s*$")


def extract_release_notes(changelog: str, version: str) -> str:
    wanted = version.strip().removeprefix("v")
    lines = changelog.splitlines()
    start: int | None = None
    end = len(lines)
    for index, line in enumerate(lines):
        match = HEADING.match(line.strip())
        if not match:
            continue
        if start is None and match.group("version") == wanted:
            start = index + 1
            continue
        if start is not None:
            end = index
            break
    if start is None:
        raise ValueError(f"CHANGELOG.md 缺少版本 {wanted} 的二级标题和更新内容。")
    body = "\n".join(lines[start:end]).strip()
    if not body or not any(line.lstrip().startswith("-") for line in body.splitlines()):
        raise ValueError(f"CHANGELOG.md 的 {wanted} 版本没有可发布的更新条目。")
    return f"## 更新内容\n\n{body}\n\n## 安装说明\n\n下载本页 Windows 安装器即可增量更新；默认保留工作区和用户数据。\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notes = extract_release_notes(args.changelog.read_text(encoding="utf-8-sig"), args.version)
    args.output.write_text(notes, encoding="utf-8")
    print(f"RELEASE_NOTES={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
