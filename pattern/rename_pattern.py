from __future__ import annotations

import argparse
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenameOp:
    src: Path
    dst: Path


def _int_basename(p: Path) -> int | None:
    try:
        return int(p.stem)
    except ValueError:
        return None


def _sorted_g_files(directory: Path, sort_mode: str) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".g"]

    if sort_mode == "mtime":
        return sorted(files, key=lambda p: p.stat().st_mtime)
    if sort_mode == "name":
        return sorted(files, key=lambda p: p.name)

    # sort_mode == "number"
    def key(p: Path):
        n = _int_basename(p)
        return (0, n) if n is not None else (1, p.name)

    return sorted(files, key=key)


def _build_ops(files: list[Path], directory: Path, expected_count: int) -> list[RenameOp]:
    if len(files) != expected_count:
        raise ValueError(f"期望有 {expected_count} 个 .g 文件，但实际是 {len(files)} 个。")

    # Use a unique prefix to avoid collisions.
    token = uuid.uuid4().hex
    temp_names = [directory / f"__tmp__rename__{token}__{i}.g" for i in range(1, expected_count + 1)]
    final_names = [directory / f"{i}.g" for i in range(1, expected_count + 1)]

    # Phase 1: original -> temp
    ops: list[RenameOp] = []
    for src, tmp in zip(files, temp_names, strict=True):
        ops.append(RenameOp(src=src, dst=tmp))

    # Phase 2: temp -> final
    for tmp, final in zip(temp_names, final_names, strict=True):
        ops.append(RenameOp(src=tmp, dst=final))

    return ops


def _validate_ops(ops: list[RenameOp], directory: Path) -> None:
    for op in ops:
        if directory not in op.src.parents and op.src != directory:
            raise ValueError(f"源文件不在目标目录内: {op.src}")
        if directory not in op.dst.parents and op.dst != directory:
            raise ValueError(f"目标文件不在目标目录内: {op.dst}")

    # Ensure temps don't already exist.
    temps = [op.dst for op in ops if "__tmp__rename__" in op.dst.name]
    conflicts = [p for p in temps if p.exists()]
    if conflicts:
        raise ValueError("临时文件名发生冲突，请重试。冲突示例: " + str(conflicts[0]))


def _run_ops(ops: list[RenameOp], dry_run: bool) -> None:
    for op in ops:
        if dry_run:
            print(f"{op.src.name} -> {op.dst.name}")
            continue

        if not op.src.exists():
            raise FileNotFoundError(f"找不到源文件: {op.src}")

        # os.replace: atomic on same filesystem, overwrites if target exists
        os.replace(op.src, op.dst)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="将目录内的 .g 文件按顺序重命名为 1.g..20.g（两阶段改名避免覆盖）"
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent),
        help="目标目录（默认：脚本所在目录）",
    )
    parser.add_argument(
        "--sort",
        choices=["number", "name", "mtime"],
        default="number",
        help="排序方式：number=按文件名数字（默认），name=按文件名，mtime=按修改时间",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="期望的 .g 文件数量（默认 20）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的重命名，不实际改名",
    )
    args = parser.parse_args(argv)

    directory = Path(args.dir).expanduser().resolve()
    if not directory.exists():
        print(f"目录不存在: {directory}", file=sys.stderr)
        return 2
    if not directory.is_dir():
        print(f"不是目录: {directory}", file=sys.stderr)
        return 2

    files = _sorted_g_files(directory, args.sort)
    try:
        ops = _build_ops(files, directory, args.count)
        _validate_ops(ops, directory)
        _run_ops(ops, dry_run=args.dry_run)
    except Exception as e:
        print(f"失败: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("dry-run 完成（未实际改名）。")
    else:
        print("重命名完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

