#!/usr/bin/env bash
# Tai nhanh cac file dataset trong download.txt bang aria2c.
# Khong tai model, llava_v1_5*.json, hoac train_val_images.zip cua TextVQA.
#
# Cach dung:
#   chmod +x download_datasets_fast.sh
#   ./download_datasets_fast.sh
#   DATA_DIR=/duong/dan/train_data ./download_datasets_fast.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_LIST="${DOWNLOAD_LIST:-$SCRIPT_DIR/download.txt}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/train_data}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-5}"
CONNECTIONS_PER_FILE="${CONNECTIONS_PER_FILE:-16}"

if ! command -v aria2c >/dev/null 2>&1; then
    printf 'Loi: chua co aria2c. Cai bang: sudo apt-get install aria2\n' >&2
    exit 1
fi

if [[ ! -f "$DOWNLOAD_LIST" ]]; then
    printf 'Loi: khong tim thay danh sach: %s\n' "$DOWNLOAD_LIST" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"
aria2_input="$(mktemp)"
trap 'rm -f -- "$aria2_input"' EXIT

in_datasets=0
selected=0

while read -r kind url destination extra; do
    if [[ "$kind" == "#datasets" ]]; then
        in_datasets=1
        continue
    fi
    if [[ "$kind" == "#models" ]]; then
        in_datasets=0
        continue
    fi

    # Chi nhan cac dong URL nam trong muc datasets; --hf (model) bi bo qua.
    [[ "$in_datasets" == "1" && "$kind" == "--url" ]] || continue
    [[ -z "${extra:-}" ]] || {
        printf 'Canh bao: bo qua dong khong hop le: %s %s %s %s\n' \
            "$kind" "$url" "$destination" "$extra" >&2
        continue
    }

    filename="${url%%\?*}"
    filename="${filename##*/}"

    # Cac file nguoi dung yeu cau loai tru.
    if [[ "$filename" == llava_v1_5*.json ]]; then
        printf 'Bo qua LLaVA JSON: %s\n' "$filename"
        continue
    fi
    if [[ "$filename" == "train_val_images.zip" && "$url" == *textvqa* ]]; then
        printf 'Bo qua TextVQA: %s\n' "$filename"
        continue
    fi

    # Giu lai cau truc thu muc nam sau thanh phan train_data trong download.txt,
    # nhung cho phep doi thu muc goc bang bien DATA_DIR.
    relative_dir="${destination#*/train_data}"
    if [[ "$relative_dir" == "$destination" ]]; then
        printf 'Canh bao: dich khong nam trong train_data, bo qua: %s\n' \
            "$destination" >&2
        continue
    fi
    output_dir="${DATA_DIR}${relative_dir}"
    mkdir -p "$output_dir"

    printf '%s\n  dir=%s\n  out=%s\n' "$url" "$output_dir" "$filename" \
        >>"$aria2_input"
    ((selected += 1))
done <"$DOWNLOAD_LIST"

if ((selected == 0)); then
    printf 'Khong co file dataset nao can tai.\n'
    exit 0
fi

printf 'Tai %d file vao %s\n' "$selected" "$DATA_DIR"
aria2c \
    --input-file="$aria2_input" \
    --continue=true \
    --max-concurrent-downloads="$PARALLEL_DOWNLOADS" \
    --max-connection-per-server="$CONNECTIONS_PER_FILE" \
    --split="$CONNECTIONS_PER_FILE" \
    --min-split-size=1M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --max-tries=10 \
    --retry-wait=3 \
    --summary-interval=5

printf 'Da tai xong %d file dataset.\n' "$selected"
