from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_json_only(
    source_dir: Path,
    destination_dir: Path,
    overwrite: bool = False,
) -> None:
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Đường dẫn nguồn không phải thư mục: {source_dir}")

    # Tránh xoá nhầm nguồn hoặc copy lồng vào chính nó
    if (
        destination_dir == source_dir
        or source_dir in destination_dir.parents
        or destination_dir in source_dir.parents
    ):
        raise ValueError(
            "Thư mục đích không được trùng, nằm trong, hoặc chứa thư mục nguồn."
        )

    if destination_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Thư mục đích đã tồn tại: {destination_dir}\n"
                "Dùng --overwrite nếu muốn xoá và copy lại."
            )
        shutil.rmtree(destination_dir)

    destination_dir.mkdir(parents=True)

    copied = 0
    for file_path in source_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            target = destination_dir / file_path.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
            copied += 1

    print("Hoàn thành.")
    print(f"Đã copy: {copied} file JSON (kể cả trong thư mục con)")
    print(f"Thư mục kết quả: {destination_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy chỉ các file JSON (đệ quy) sang thư mục đích."
    )
    parser.add_argument("source_dir", type=Path, help="Thư mục nguồn.")
    parser.add_argument("destination_dir", type=Path, help="Thư mục đích.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xoá thư mục đích nếu đã tồn tại rồi copy lại.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        copy_json_only(args.source_dir, args.destination_dir, args.overwrite)
    except (
        FileNotFoundError,
        NotADirectoryError,
        FileExistsError,
        PermissionError,
        ValueError,
    ) as error:
        print(f"Lỗi: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()