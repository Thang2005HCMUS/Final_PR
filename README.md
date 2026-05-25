# Đồ án cuối kì môn Nhận dạng 
Hướng dẫn sử dụng 
### Tải dataset 
**Ubuntu or Colab**
```
chmod +x ./download_dataset.sh
./download_dataset.sh
```

### Tải model weights 
**Ubuntu or Colab**
```
chmod +x ./download_model.sh
./download_model.sh
```
### Cấu hình chạy eval 
chỉnh ```config/config_eval.py``` các thuộc tính sau 

- ```config.model_name``` : tên model , có thể là baseline hoặc clip hoặc dinov2

- ```config.model_path``` : đường dẫn đến file trọng số

- ```config.eval_path``` : đường dẫn đến folder chứa tập val 

- ```config.val_targets``` : danh sách dataset được dùng để eval

### Chạy Eval 
mở terminal gõ 
```
python src/eval.py
```
file eval.py sẽ load full dataset vào RAM , nếu máy ít RAM thì dùng 
```
python src/eval_optimized.py
```
### Chạy Eval trên tất cả các model 1 lần 
Tìm đến ```python src/eval_all.py```
sửa 
- ```DATASETS``` :danh sách các dataset dùng để eval
- ```MODELS_TO_EVAL``` : danh sách các model được dùng để eval 
```
python src/eval_all.py
```

