#!/usr/bin/env bash
# Download and extract the datasets used for VLM training.
#
# Usage:
#   chmod +x download_datatrain.sh
#   ./download_datatrain.sh
#
# Options:
#   FORCE_REDOWNLOAD=1 ./download_datatrain.sh  # Download all data again
#   KEEP_ARCHIVES=1   ./download_datatrain.sh  # Keep ZIP files after extraction
#   DATA_DIR=/path/to/train_data ./download_datatrain.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/train_data}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="$LOG_DIR/download_$(date '+%Y%m%d_%H%M%S').log"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-0}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

mkdir -p "$LOG_DIR"
# Display output in the terminal and save stdout/stderr to a log file.
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    local level="$1"
    shift
    printf '[%s] [%-5s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$*"
}

die() {
    log ERROR "$*"
    exit 1
}

for command_name in wget unzip tee find sort; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "Thiếu lệnh '$command_name'. Hãy cài đặt rồi chạy lại."
done

mkdir -p "$DATA_DIR"/{coco,gqa,ocr_vqa/images,textvqa,vg}
cd "$DATA_DIR" || die "Không thể truy cập: $DATA_DIR"

is_nonempty_dir() {
    local directory="$1"
    [[ -d "$directory" ]] && [[ -n "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]
}

download_file() {
    local name="$1" url="$2" output="$3"

    mkdir -p "$(dirname "$output")"
    if [[ -s "$output" && "$FORCE_REDOWNLOAD" != "1" ]]; then
        log SKIP "[$name] Đã tồn tại: $output"
        return 0
    fi

    [[ "$FORCE_REDOWNLOAD" == "1" ]] && rm -f -- "$output"
    log INFO "[$name] Bắt đầu tải: $output"
    if wget --continue --progress=bar:force:noscroll --tries=5 --timeout=30 \
        --output-document="$output" "$url"; then
        log DONE "[$name] Tải xong: $output"
    else
        log ERROR "[$name] Tải thất bại: $url (giữ file dở để tiếp tục lần sau)"
        return 1
    fi
}

process_zip() {
    local name="$1" url="$2" archive="$3" destination="$4" check_dir="$5"

    if is_nonempty_dir "$check_dir" && [[ "$FORCE_REDOWNLOAD" != "1" ]]; then
        log SKIP "[$name] Dữ liệu đã có trong: $check_dir"
        return 0
    fi

    if [[ -s "$archive" && "$FORCE_REDOWNLOAD" != "1" ]]; then
        log INFO "[$name] Dùng file đã tải: $archive"
    else
        download_file "$name" "$url" "$archive" || return 1
    fi

    log INFO "[$name] Đang kiểm tra file zip..."
    if ! unzip -tq "$archive" >/dev/null; then
        log ERROR "[$name] File zip không hợp lệ hoặc chưa tải đủ: $archive"
        return 1
    fi

    mkdir -p "$destination"
    log INFO "[$name] Đang giải nén: $archive -> $destination"
    if ! unzip -q -o "$archive" -d "$destination"; then
        log ERROR "[$name] Giải nén thất bại; file zip được giữ lại."
        return 1
    fi

    if [[ "$KEEP_ARCHIVES" == "1" ]]; then
        log DONE "[$name] Giải nén xong; giữ lại $archive"
    else
        rm -f -- "$archive"
        log DONE "[$name] Giải nén xong; đã xóa file zip."
    fi
}

log INFO "Thư mục dữ liệu: $DATA_DIR"
log INFO "File log: $LOG_FILE"
log INFO "Bắt đầu tải song song. Tiến trình sẽ hiển thị ngay trên terminal."

declare -a job_pids=()
declare -a job_names=()

start_job() {
    local name="$1"
    shift
    "$@" &
    job_pids+=("$!")
    job_names+=("$name")
}

start_job "LLaVA" download_file "LLaVA" \
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json" \
    "llava_v1_5_mix665k.json"

start_job "OCR-VQA metadata" download_file "OCR-VQA metadata" \
    "https://huggingface.co/datasets/DVLe/ocr_vqa/resolve/main/dataset.json" \
    "ocr_vqa/dataset.json"

start_job "COCO" process_zip "COCO" \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "coco/train2017.zip" "coco" "coco/train2017"

start_job "GQA" process_zip "GQA" \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
    "gqa/images.zip" "gqa" "gqa/images"

start_job "TextVQA" process_zip "TextVQA" \
    "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
    "textvqa/train_val_images.zip" "textvqa" "textvqa/train_images"

start_job "Visual Genome 1" process_zip "Visual Genome 1" \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
    "vg/images.zip" "vg" "vg/VG_100K"

start_job "Visual Genome 2" process_zip "Visual Genome 2" \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    "vg/images2.zip" "vg" "vg/VG_100K_2"

failed_jobs=0
for index in "${!job_pids[@]}"; do
    if wait "${job_pids[$index]}"; then
        log DONE "Job hoàn tất: ${job_names[$index]}"
    else
        log ERROR "Job thất bại: ${job_names[$index]}"
        ((failed_jobs += 1))
    fi
done

log INFO "Xử lý ảnh OCR-VQA..."
if is_nonempty_dir "ocr_vqa/images" && [[ "$FORCE_REDOWNLOAD" != "1" ]]; then
    log SKIP "OCR-VQA images đã có dữ liệu."
elif [[ -f "ocr_vqa/loadDataset.py" ]]; then
    log INFO "Chạy ocr_vqa/loadDataset.py"
    if ! (cd ocr_vqa && python3 loadDataset.py); then
        log ERROR "Tải ảnh OCR-VQA thất bại."
        ((failed_jobs += 1))
    fi
else
    log WARN "Không tìm thấy ocr_vqa/loadDataset.py; chưa thể tải ảnh OCR-VQA."
    log WARN "Hãy đặt file từ repo OCR-VQA-200K vào: $DATA_DIR/ocr_vqa/loadDataset.py"
fi

log INFO "Cấu trúc dữ liệu hiện tại:"
find "$DATA_DIR" -maxdepth 2 -print | sort

if ((failed_jobs > 0)); then
    die "Có $failed_jobs job thất bại. Xem chi tiết tại: $LOG_FILE"
fi

log DONE "Hoàn tất. Log được lưu tại: $LOG_FILE"
