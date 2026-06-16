import os
import sys
import pickle
import numpy as np
from PIL import Image

# Đảm bảo import được các thư viện nội bộ nếu cần
sys.path.append(os.path.join(os.getcwd()))

BIN_PATH = "eval_datasets/lfw.bin"
GALLERY_DIR = "my_gallery"
TEST_DIR = "my_test"

os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print("=== ĐANG ĐỌC FILE LFW.BIN ===")
if not os.path.exists(BIN_PATH):
    print(f"Không tìm thấy file {BIN_PATH}. Vui lòng kiểm tra lại đường dẫn!")
    sys.exit()

# Đọc file .bin chứa mã hóa ảnh giống trong eval_all.py
try:
    with open(BIN_PATH, 'rb') as f:
        bins, issame_list = pickle.load(f)
except UnicodeDecodeError:
    with open(BIN_PATH, 'rb') as f:
        bins, issame_list = pickle.load(f, encoding='bytes')

print(f"Tìm thấy tổng cộng {len(bins)} ảnh ({len(issame_list)} cặp ảnh đối chiếu).")
print("Đang trích xuất và phân tách thành Database (Gallery) & Test...")

import mxnet as mx  # Dùng mxnet giải mã ảnh giống code gốc của bạn

# Quy trình tách: 
# Chúng ta sẽ quét qua các cặp ảnh được xác định là CÙNG 1 NGƯỜI (issame = True)
# Ảnh thứ nhất của cặp sẽ đưa vào Gallery (Làm ảnh gốc trong Database)
# Ảnh thứ hai của cặp sẽ đưa vào thư mục Test (Dùng để nạp vào GUI test)

person_id = 0
for idx, issame in enumerate(issame_list):
    if issame: # Chỉ xử lý các cặp là cùng một người
        # Lấy index của 2 ảnh trong mảng bins (mỗi cặp gồm 2 ảnh liên tiếp: idx*2 và idx*2 + 1)
        idx_img0 = idx * 2
        idx_img1 = idx * 2 + 1
        
        # Giải mã ảnh từ bytes bằng mxnet giống trong OptimizedBinLoader
        img0_mx = mx.image.imdecode(bins[idx_img0])
        img1_mx = mx.image.imdecode(bins[idx_img1])
        
        # Chuyển sang dạng numpy và lưu bằng PIL Image
        img0_np = img0_mx.asnumpy()
        img1_np = img1_mx.asnumpy()
        
        im0 = Image.fromarray(img0_np)
        im1 = Image.fromarray(img1_np)
        
        # Đặt tên định danh theo số ID (Ví dụ: Person_0001, Person_0002...)
        # Ảnh 0 làm Database định danh
        im0.save(os.path.join(GALLERY_DIR, f"Person_{person_id:04d}.jpg"))
        # Ảnh 1 làm ảnh đem đi Test xem AI có nhận diện ra đúng ID đó không
        im1.save(os.path.join(TEST_DIR, f"test_Person_{person_id:04d}.jpg"))
        
        person_id += 1
        
        # Giới hạn trích xuất khoảng 200 người để test cho nhẹ máy, tránh tràn ổ cứng
        if person_id >= 200:
            break

print("----------------------------------------------------------------")
print(f"✅ HOÀN THÀNH TRÍCH XUẤT!")
print(f"- Đã tạo {len(os.listdir(GALLERY_DIR))} người mẫu trong Database: '{GALLERY_DIR}/'")
print(f"- Đã tạo {len(os.listdir(TEST_DIR))} ảnh thách thức tương ứng trong tập Test: '{TEST_DIR}/'")
print("Giờ bạn có thể mở GUI lên, nạp thư mục 'my_gallery' và dùng ảnh trong 'my_test' để kiểm tra rồi!")