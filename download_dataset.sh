#!/bin/bash

# Khai báo tên thư mục
TARGET_DIR="eval_datasets"

echo "=========================================="
echo "BẮT ĐẦU TẢI DỮ LIỆU ĐÁNH GIÁ (EVAL DATASETS)"
echo "=========================================="

# Tạo thư mục nếu chưa có
echo "[1/3] Đang tạo thư mục: $TARGET_DIR..."
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit

echo "[2/3] Đang tải các file .bin từ HuggingFace..."

# Lệnh tải kèm thanh tiến trình (-q --show-progress) và bỏ qua nếu file đã tồn tại (-nc)
wget -nc -q --show-progress https://huggingface.co/datasets/Icar/val_sets/resolve/main/lfw.bin
wget -nc -q --show-progress https://huggingface.co/datasets/Icar/val_sets/resolve/main/cfp_fp.bin
wget -nc -q --show-progress https://huggingface.co/datasets/Icar/val_sets/resolve/main/agedb_30.bin
wget -nc -q --show-progress https://huggingface.co/datasets/Icar/val_sets/resolve/main/calfw.bin
wget -nc -q --show-progress https://huggingface.co/datasets/Icar/val_sets/resolve/main/cplfw.bin

echo "=========================================="
echo "[3/3] TẢI HOÀN TẤT!"
echo "Đường dẫn thư mục hiện tại: $(pwd)"
echo "Danh sách các file:"
ls -lh
echo "=========================================="