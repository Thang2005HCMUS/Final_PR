import os
import sys
import torch
import pickle
import logging
import mxnet as mx
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.getcwd()))

from config.config_eval import config as cfg
from backbone import get_model
from data.transform import transform_image
from utils.logging import init_logging
from utils.evaluation import evaluate

# =====================================================================
# CẤU HÌNH DANH SÁCH DATASETS TRÊN MÁY CỦA BẠN
# =====================================================================
DATASETS = {
    "lfw": "eval_datasets/lfw.bin",
    "cfp_fp": "eval_datasets/cfp_fp.bin",
    "cfp_ff": "eval_datasets/cfp_ff.bin",
    "agedb_30": "eval_datasets/agedb_30.bin",
    "calfw": "eval_datasets/calfw.bin",
    "cplfw": "eval_datasets/cplfw.bin"
}

# =====================================================================
# DANH SÁCH TOÀN BỘ 18 MODELS CÓ TRONG PAPER (MỞ/CÓ THỂ COMMENT HÓA)
# =====================================================================
MODELS_TO_EVAL = {
    # --- 1. BASELINE MODELS (Mô hình huấn luyện từ đầu - Tiêu chuẩn) ---
    # Baseline - Small (Sử dụng cấu hình backbone: small)
    "Baseline_Small_CASIA-WebFace": "weights/Baseline/Small/Baseline_Small_CASIA-WebFace.pth",
    "Baseline_Small_MS1MV2": "weights/Baseline/Small/Baseline_Small_MS1MV2.pth",
    "Baseline_Small_WebFace4M": "weights/Baseline/Small/Baseline_Small_WebFace4M.pth",
    
    # Baseline - Large (Sử dụng cấu hình backbone: large)
    # "Baseline_Large_CASIA-WebFace": "weights/Baseline/Large/Baseline_Large_CASIA-WebFace.pth",
    # "Baseline_Large_MS1MV2": "weights/Baseline/Large/Baseline_Large_MS1MV2.pth",
    # "Baseline_Large_WebFace4M": "weights/Baseline/Large/Baseline_Large_WebFace4M.pth",


    # --- 2. CLIP BASED MODELS (Mô hình dựa trên Foundation Model CLIP) ---
    # CLIP - Base (Sử dụng cấu hình backbone: ViT-B/16)
    # "CLIP_Base_CASIA-WebFace": "weights/CLIP/Base/CLIP_Base_CASIA-WebFace.pth",
    # "CLIP_Base_MS1MV2": "weights/CLIP/Base/CLIP_Base_MS1MV2.pth",
    # "CLIP_Base_WebFace4M": "weights/CLIP/Base/CLIP_Base_WebFace4M.pth",
    
    # CLIP - Large (Sử dụng cấu hình backbone: ViT-L/14)
    # "CLIP_Large_CASIA-WebFace": "weights/CLIP/Large/CLIP_Large_CASIA-WebFace.pth",
    # "CLIP_Large_MS1MV2": "weights/CLIP/Large/CLIP_Large_MS1MV2.pth",
    # "CLIP_Large_WebFace4M": "weights/CLIP/Large/CLIP_Large_WebFace4M.pth",


    # --- 3. DINOv2 BASED MODELS (Mô hình dựa trên Foundation Model DINOv2) ---
    # DINOv2 - Small (Sử dụng cấu hình backbone: small)
    # "DINOv2_Small_CASIA-WebFace": "weights/DINOv2/Small/DINOv2_Small_CASIA-WebFace.pth",
    # "DINOv2_Small_MS1MV2": "weights/DINOv2/Small/DINOv2_Small_MS1MV2.pth",
    # "DINOv2_Small_WebFace4M": "weights/DINOv2/Small/DINOv2_Small_WebFace4M.pth",
    
    # DINOv2 - Base (Sử dụng cấu hình backbone: base)
    # "DINOv2_Base_CASIA-WebFace": "weights/DINOv2/Base/DINOv2_Base_CASIA-WebFace.pth",
    # "DINOv2_Base_MS1MV2": "weights/DINOv2/Base/DINOv2_Base_MS1MV2.pth",
    # "DINOv2_Base_WebFace4M": "weights/DINOv2/Base/DINOv2_Base_WebFace4M.pth",
}

# =====================================================================
# LỚP ĐỌC FILE .BIN THEO BATCH (BẢO VỆ RAM TUYỆT ĐỐI)
# =====================================================================
class OptimizedBinLoader:
    def __init__(self, bin_path, image_size, transform):
        self.image_size = (image_size, image_size)
        self.transform = transform
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Dataset file not found: {bin_path}")
            
        try:
            with open(bin_path, 'rb') as f:
                self.bins, self.issame_list = pickle.load(f)
        except UnicodeDecodeError:
            with open(bin_path, 'rb') as f:
                self.bins, self.issame_list = pickle.load(f, encoding='bytes')
        
        self.num_samples = len(self.bins)

    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, self.num_samples)
        count = end_idx - start_idx
        
        batch_data0 = torch.empty((count, 3, self.image_size[0], self.image_size[1]))
        batch_data1 = torch.empty((count, 3, self.image_size[0], self.image_size[1]))
        
        for i, idx in enumerate(range(start_idx, end_idx)):
            _bin = self.bins[idx]
            img = mx.image.imdecode(_bin)
            if img.shape[1] != self.image_size[0]:
                img = mx.image.resize_short(img, self.image_size[0])
            
            img_np = img.asnumpy()
            img_transformed = self.transform(img_np)
            img_mx = mx.nd.array(img_transformed)
            
            batch_data0[i] = torch.from_numpy(img_mx.asnumpy())
            img_flip = mx.ndarray.flip(data=img_mx, axis=2)
            batch_data1[i] = torch.from_numpy(img_flip.asnumpy())
            
        return batch_data0, batch_data1

@torch.no_grad()
def run_eval_dataset(bin_path, backbone, batch_size, image_size, transform):
    loader = OptimizedBinLoader(bin_path, image_size, transform)
    embeddings_list0 = []
    embeddings_list1 = []
    num_samples = loader.num_samples
    
    print(f"      -> Total images: {num_samples}. Processing with batch size: {batch_size}...")
    
    for ba in range(0, num_samples, batch_size):
        data0, data1 = loader.get_batch(ba, batch_size)
        img0, img1 = data0.to("cuda"), data1.to("cuda")
        
        out0 = backbone(img0)
        out1 = backbone(img1)
        
        # Sửa lỗi AttributeError: Kiểm tra thông minh đầu ra để lấy vector đặc trưng chính xác
        if hasattr(out0, 'pooler_output'):
            out0 = out0.pooler_output
            out1 = out1.pooler_output
            
        embeddings_list0.append(out0.detach().cpu().numpy())
        embeddings_list1.append(out1.detach().cpu().numpy())
        
        # IN TIẾN TRÌNH TỪNG BATCH: Cứ sau mỗi 5 batch hoặc batch cuối cùng thì print báo cáo
        if ba % (batch_size * 5) == 0 or ba + batch_size >= num_samples:
            current_processed = min(ba + batch_size, num_samples)
            print(f"      [VRAM Progress] Inference progress: {current_processed}/{num_samples}")
        
        del data0, data1, img0, img1
        torch.cuda.empty_cache()

    embeddings0 = np.vstack(embeddings_list0)
    embeddings1 = np.vstack(embeddings_list1)
    
    import sklearn.preprocessing as preprocessing
    embeddings = preprocessing.normalize(embeddings0) + preprocessing.normalize(embeddings1)
    embeddings = preprocessing.normalize(embeddings)
    
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, loader.issame_list, nrof_folds=10)
    
    acc_flip, std_flip = np.mean(accuracy), np.std(accuracy)
    
    # Tính thêm XNorm (Độ dài vector đặc trưng) giống mã nguồn gốc
    xnorm = np.mean(np.linalg.norm(embeddings, axis=1))
    
    del loader, embeddings_list0, embeddings_list1, embeddings0, embeddings1
    import gc
    gc.collect()
    
    # Trả về toàn bộ các chỉ số thu hoạch được
    return {
        "accuracy": acc_flip,
        "std": std_flip,
        "xnorm": xnorm,
        "val_rate": val,
        "val_std": val_std,
        "far": far
    }
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Activated Device: {device}")
    
    transform = transform_image(
        image_size=cfg.image_size, 
        normalize_type=cfg.normalize_type,
        interpolation_type=cfg.interpolation_type
    )

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(logging.getLogger(), 0, cfg.output, logfile="batch_eval_summary.log")

    # VÒNG LẶP CHÍNH: QUÉT QUA CÁC MODEL KHÔNG BỊ COMMENT
    for model_key, model_weight_path in MODELS_TO_EVAL.items():
        if not os.path.exists(model_weight_path):
            print(f"\n[SKIP] Weight file not found for {model_key} at {model_weight_path}")
            continue
            
        print(f"\n==================================================================")
        print(f"🔥 START EVALUATION FOR MODEL: {model_key}")
        print(f"==================================================================")

        # TỰ ĐỘNG KHỚP CẤU HÌNH BACKBONE THEO TÊN MODEL KEY ĐỂ GET_MODEL CHÍNH XÁC
        model_name_lower = model_key.lower()
        if "clip" in model_name_lower:
            cfg.model_name = "clip"
            cfg.backbone_size = "ViT-B/16" if "base" in model_name_lower else "ViT-L/14"
        elif "dinov2" in model_name_lower:
            cfg.model_name = "dinov2"
            cfg.backbone_size = "small" if "small" in model_name_lower else "base"
        else:
            cfg.model_name = "baseline"
            cfg.backbone_size = "small" if "small" in model_name_lower else "large"

        # Khởi tạo mô hình tương ứng
        model = get_model(0, **cfg)
        
        # Tải trọng số an toàn lên GPU
        print(f"-> Loading weights from: {model_weight_path}")
        state_dict = torch.load(model_weight_path, map_location=device)
        model.backbone.load_state_dict(state_dict)
        backbone_net = model.backbone.to(device)
        backbone_net.eval()

        # Tạo cấu trúc folder độc lập: <output_dir>/<tên_model>/
        model_output_dir = os.path.join(cfg.output, model_key)
        os.makedirs(model_output_dir, exist_ok=True)

        # VÒNG LẶP CON: EVALUATE LẦN LƯỢT TỪNG DATASET TRÊN MODEL HIỆN TẠI
        for dataset_name, dataset_bin_path in DATASETS.items():
            if not os.path.exists(dataset_bin_path):
                print(f"   [WARNING] Dataset {dataset_name} not found, Skipping...")
                continue
                
            print(f"   ⚡ Dataset: {dataset_name} ...")
            
            # Khởi chạy hàm đánh giá theo batch tiết kiệm RAM
            res = run_eval_dataset(
                dataset_bin_path, backbone_net, cfg.batch_size_eval, cfg.image_size, transform
            )
            
            # Ghi file kết quả .txt độc lập cho từng cặp
            result_file_path = os.path.join(model_output_dir, f"{dataset_name}.txt")
            with open(result_file_path, "w", encoding="utf-8") as f_out:
                f_out.write(f"=========================================\n")
                f_out.write(f" MODEL   : {model_key}\n")
                f_out.write(f" DATASET : {dataset_name}\n")
                f_out.write(f"=========================================\n")
                f_out.write(f" 1. Accuracy (Flip)      : {res['accuracy']:.5f} +- {res['std']:.5f}\n")
                f_out.write(f" 2. Verification Rate     : {res['val_rate']:.5f} +- {res['val_std']:.5f}\n")
                f_out.write(f" 3. FAR (False Accept)    : {res['far']:.5f}\n")
                f_out.write(f" 4. XNorm (Embedding Length): {res['xnorm']:.5f}\n")
            
            # Đồng thời bắn vào file log tổng hợp hệ thống
            logging.info(f"[{model_key}][{dataset_name}] Acc: {res['accuracy']:.5f} | ValRate: {res['val_rate']:.5f} | XNorm: {res['xnorm']:.5f}")
            print(f"   ✅ Finished [{dataset_name}] -> Detailed results saved to {result_file_path}")
            
        # Giải phóng triệt để RAM/VRAM của mô hình hiện tại trước khi đổi sang mô hình mới
        del model, backbone_net, state_dict
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    print("\n🎉🎉🎉 BATCH EVALUATION COMPLETED FOR ALL SELECTED MODELS! 🎉🎉🎉")

if __name__ == "__main__":
    main()