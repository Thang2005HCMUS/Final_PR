import logging
import os
import sys
import torch
import argparse
import pickle
import mxnet as mx
import numpy as np

sys.path.append(os.path.join(os.getcwd()))

from config.config_eval import config as cfg
from backbone import get_model
from data.transform import transform_image
from utils.logging import init_logging
from finetuning import apply_lora_model
from utils.evaluation import evaluate

# --- CLASS TỐI ƯU BỘ NHỚ: ĐỌC ẢNH THEO BATCH TRỰC TIẾP TỪ FILE BIN ---
class OptimizedBinLoader:
    def __init__(self, bin_path, image_size, transform):
        self.image_size = (image_size, image_size)
        self.transform = transform
        
        # Chỉ load danh sách byte thô và nhãn (rất nhẹ, vài trăm MB)
        try:
            with open(bin_path, 'rb') as f:
                self.bins, self.issame_list = pickle.load(f)
        except UnicodeDecodeError:
            with open(bin_path, 'rb') as f:
                self.bins, self.issame_list = pickle.load(f, encoding='bytes')
        
        self.num_samples = len(self.bins)

    def get_batch(self, start_idx, batch_size):
        """Hàm này chỉ giải nén ĐÚNG số lượng ảnh trong 1 batch rồi nạp vào RAM"""
        end_idx = min(start_idx + batch_size, self.num_samples)
        count = end_idx - start_idx
        
        # Khởi tạo tensor trống cho batch hiện tại (gồm ảnh gốc và ảnh lật)
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
            
            # Ảnh gốc
            batch_data0[i] = torch.from_numpy(img_mx.asnumpy())
            # Ảnh lật ngang
            img_flip = mx.ndarray.flip(data=img_mx, axis=2)
            batch_data1[i] = torch.from_numpy(img_flip.asnumpy())
            
        return batch_data0, batch_data1

@torch.no_grad()
def run_optimized_test(bin_path, backbone, model_name, batch_size, image_size, transform):
    """Hàm chạy test chia nhỏ batch, tính xong là giải phóng RAM ngay"""
    loader = OptimizedBinLoader(bin_path, image_size, transform)
    
    embeddings_list0 = []
    embeddings_list1 = []
    
    num_samples = loader.num_samples
    print(f"Total samples to process: {num_samples}. Processing in batches of {batch_size}...")
    
    # Duyệt theo batch size
    for ba in range(0, num_samples, batch_size):
        # 1. Chỉ giải nén đúng 1 batch ảnh đưa vào RAM
        data0, data1 = loader.get_batch(ba, batch_size)
        
        # 2. Đẩy lên GPU tính toán luôn
        img0 = data0.to("cuda")
        img1 = data1.to("cuda")
        
        out0 = backbone(img0)
        out1 = backbone(img1)
        
        # Nếu đầu ra là Object có pooler_output (như một số bản HuggingFace) thì lấy thuộc tính đó, ngược lại lấy trực tiếp Tensor
        if hasattr(out0, 'pooler_output'):
            out0 = out0.pooler_output
            out1 = out1.pooler_output
            
        embeddings_list0.append(out0.detach().cpu().numpy())
        embeddings_list1.append(out1.detach().cpu().numpy())
        
        if ba % (batch_size * 5) == 0 or ba + batch_size >= num_samples:
            print(f"Processed embeddings: {min(ba + batch_size, num_samples)}/{num_samples}")
            
        # 3. Ép giải phóng bộ nhớ ảnh thô ngay trong vòng lặp
        del data0, data1, img0, img1
        torch.cuda.empty_cache()

    # Gom các cụm embedding lại (đống này chỉ là vector số nên cực kỳ nhẹ)
    embeddings0 = np.vstack(embeddings_list0)
    embeddings1 = np.vstack(embeddings_list1)
    
    # Tính toán chuẩn hóa và cộng gộp đặc trưng ảnh gốc + ảnh lật
    import sklearn.preprocessing as preprocessing
    embeddings = preprocessing.normalize(embeddings0) + preprocessing.normalize(embeddings1)
    embeddings = preprocessing.normalize(embeddings)
    
    # Gọi hàm evaluate gốc của project để tính toán toán học (ROC/Accuracy)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, loader.issame_list, nrof_folds=10)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    
    return acc2, std2

def evaluate_optimized(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    print(f"Using device: {device}")

    log_root = logging.getLogger()
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    init_logging(log_root, local_rank, cfg.output, logfile=cfg.log_name)

    transform = transform_image(
        image_size=cfg.image_size, 
        normalize_type=cfg.normalize_type,
        interpolation_type=cfg.interpolation_type
    )

    # Khởi tạo mô hình
    model = get_model(local_rank, **cfg)
    if cfg.use_lora:
        apply_lora_model(local_rank, model, **cfg)

    if cfg.model_path:
        print(f"Loading model from path: {cfg.model_path}")
        state_dict = torch.load(cfg.model_path, map_location=device)
        model.backbone.load_state_dict(state_dict)

    model = model.backbone.to(device)
    model.eval()
    
    print("Starting optimized evaluation...")
    
    # Lần lượt duyệt qua từng file target để giải phóng RAM triệt để
    for name in cfg.val_targets:
        bin_path = os.path.join(cfg.eval_path, name + ".bin")
        if os.path.exists(bin_path):
            print(f"\n==== Evaluating: {name} ====")
            acc, std = run_optimized_test(
                bin_path, model, cfg.model_name, cfg.batch_size_eval, cfg.image_size, transform
            )
            logging.info('[%s] Accuracy-Flip: %1.5f+-%1.5f' % (name, acc, std))
            print(f"Result [{name}] -> Accuracy: {acc:.5f} +- {std:.5f}")
        else:
            print(f"File not found: {bin_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized Evaluation')
    args = parser.parse_args()
    evaluate_optimized(args)