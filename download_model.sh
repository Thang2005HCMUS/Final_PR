#!/bin/bash

echo "======================================================"
echo " BẮT ĐẦU TẢI PRETRAINED MODELS (FROUNDATION)"
echo " Nguồn: OwnCloud Fraunhofer (Đã tự động thêm /download)"
echo "======================================================"

# Hàm tải file và lưu vào thư mục phân cấp
download_model() {
    local url=$1
    local method=$2
    local backbone=$3
    local data=$4
    local filename="${method}_${backbone}_${data}.pth"
    local dir_path="weights/${method}/${backbone}"
    
    # Thêm hậu tố /download để lấy direct link
    local direct_url="${url}/download"

    # Tạo thư mục
    mkdir -p "$dir_path"
    
    echo ">> Đang tải: $filename"
    echo ">> Lưu tại: $dir_path/"
    
    # Tải bằng wget (thêm --content-disposition để bắt đúng tên file nếu server có trả về, 
    # nhưng ở đây ta chỉ định luôn tên output bằng -O cho dễ quản lý)
    wget -q --show-progress -nc -O "${dir_path}/${filename}" "$direct_url"
    
    echo "------------------------------------------------------"
}

# 1. BASELINE MODELS
echo -e "\n--- TẢI BASELINE MODELS ---"

# Baseline - Small
download_model "https://owncloud.fraunhofer.de/index.php/s/RqDWPal8qvXKuH7" "Baseline" "Small" "CASIA-WebFace"
download_model "https://owncloud.fraunhofer.de/index.php/s/7HubejKcXwhxk8D" "Baseline" "Small" "MS1MV2"
download_model "https://owncloud.fraunhofer.de/index.php/s/ZXnzAODROxWncil" "Baseline" "Small" "WebFace4M"

# Baseline - Large
# download_model "https://owncloud.fraunhofer.de/index.php/s/bugYxj2g6lmbtbZ" "Baseline" "Large" "CASIA-WebFace"
# download_model "https://owncloud.fraunhofer.de/index.php/s/AeaxyTB1j6U2Nb5" "Baseline" "Large" "MS1MV2"
# download_model "https://owncloud.fraunhofer.de/index.php/s/XGFtvGLFM39aPtV" "Baseline" "Large" "WebFace4M"


# 2. CLIP MODELS
echo -e "\n--- TẢI CLIP MODELS ---"

# CLIP - Base
download_model "https://owncloud.fraunhofer.de/index.php/s/oeyZTsXKYKFID5M" "CLIP" "Base" "CASIA-WebFace"
download_model "https://owncloud.fraunhofer.de/index.php/s/OyD0N0KYvyToBBr" "CLIP" "Base" "MS1MV2"
download_model "https://owncloud.fraunhofer.de/index.php/s/p2ZbWOsp1nVLv0f" "CLIP" "Base" "WebFace4M"

# CLIP - Large
# download_model "https://owncloud.fraunhofer.de/index.php/s/tkz0g8og3pKpISd" "CLIP" "Large" "CASIA-WebFace"
# download_model "https://owncloud.fraunhofer.de/index.php/s/fYKGSSA1HviGIBo" "CLIP" "Large" "MS1MV2"
# download_model "https://owncloud.fraunhofer.de/index.php/s/w9w0Sa4vnDYd79y" "CLIP" "Large" "WebFace4M"

# # 3. DINOv2 MODELS
echo -e "\n--- DINOv2 MODELS (Hiện tại README chưa cập nhật link) ---"

echo "======================================================"
echo " HOÀN TẤT TẢI XUỐNG!"
echo " Kiểm tra các file đã tải trong thư mục: ./weights/"
echo "======================================================"