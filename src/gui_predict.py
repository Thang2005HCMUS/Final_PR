import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import numpy as np
import mxnet as mx

# Đảm bảo thêm đường dẫn hệ thống để import chuẩn bộ nguồn
sys.path.append(os.path.join(os.getcwd()))

from config.config_eval import config as cfg
from backbone import get_model
from data.transform import transform_image

class FaceRecognitionGUI:
    # ── màu sắc & font ──────────────────────────────────────────────
    BG          = "#0F1117"   # nền tối
    PANEL       = "#1A1D27"   # card
    BORDER      = "#2A2D3E"   # viền
    ACCENT      = "#4F8EF7"   # xanh chủ đạo
    ACCENT2     = "#7C5CFC"   # tím nhấn
    SUCCESS     = "#2ECC71"
    DANGER      = "#E74C3C"
    TEXT        = "#E8EAF0"
    SUBTEXT     = "#8890A8"
    FONT_FAMILY = "Roboto"    # Đổi cứng thành Roboto chuẩn trên Linux

    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1050x680")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)

        # Biến lưu đường dẫn giao diện
        self.selected_model_path = tk.StringVar()
        self.selected_db_dir = tk.StringVar()
        self.selected_query_img = tk.StringVar()

        # Cấu hình phần cứng độc lập
        self.backbone_net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_fonts()
        self._setup_styles()
        self.create_widgets()

    def _setup_fonts(self):
        """Định nghĩa font chính xác dựa trên fc-list của hệ thống Linux"""
        # Sử dụng 'Roboto' - font tĩnh hiển thị cực đẹp và mượt trên Linux X11
        chosen = "Roboto"
        self.FONT_FAMILY = chosen

        self.f_title  = (chosen, 14, "bold")
        self.f_label  = (chosen, 10)
        self.f_small  = (chosen, 9)
        self.f_result = (chosen, 14, "bold")
        self.f_score  = (chosen, 11)
        self.f_btn    = (chosen, 10, "bold")

    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        BG, PANEL, BORDER = self.BG, self.PANEL, self.BORDER
        ACCENT, TEXT, SUBTEXT = self.ACCENT, self.TEXT, self.SUBTEXT

        style.configure(".",
            background=BG, foreground=TEXT,
            font=self.f_label, borderwidth=0)

        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL, relief="flat")

        style.configure("TLabel",
            background=BG, foreground=TEXT, font=self.f_label)
        style.configure("Card.TLabel",
            background=PANEL, foreground=TEXT, font=self.f_label)
        style.configure("Sub.TLabel",
            background=PANEL, foreground=SUBTEXT, font=self.f_small)

        # Entry
        style.configure("TEntry",
            fieldbackground="#252836", foreground=TEXT,
            insertcolor=TEXT, borderwidth=1, relief="flat")
        style.map("TEntry", bordercolor=[("focus", ACCENT), ("!focus", BORDER)])

        # Nút chính (Load)
        style.configure("Accent.TButton",
            background=ACCENT, foreground="#FFFFFF",
            font=self.f_btn, borderwidth=0, padding=(14, 8))
        style.map("Accent.TButton",
            background=[("active", "#3A7BE8"), ("disabled", "#2A2D3E")],
            foreground=[("disabled", SUBTEXT)])

        # Nút phụ (Browse)
        style.configure("Ghost.TButton",
            background=PANEL, foreground=ACCENT,
            font=self.f_label, borderwidth=1, relief="flat", padding=(10, 6))
        style.map("Ghost.TButton",
            background=[("active", BORDER)])

        # Nút Predict
        style.configure("Predict.TButton",
            background=self.ACCENT2, foreground="#FFFFFF",
            font=(self.FONT_FAMILY, 11, "bold"), borderwidth=0, padding=(14, 10))
        style.map("Predict.TButton",
            background=[("active", "#6A4DE8"), ("disabled", "#2A2D3E")],
            foreground=[("disabled", SUBTEXT)])

    # ── helper: tạo card có border ─────────────────────────────────
    def _card(self, parent, **kw):
        outer = tk.Frame(parent, bg=self.BORDER, bd=0)
        inner = tk.Frame(outer, bg=self.PANEL, bd=0)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return outer, inner

    def _section_title(self, parent, text):
        tk.Label(parent, text=text,
                 bg=self.PANEL, fg=self.ACCENT,
                 font=self.f_title).pack(anchor="w", pady=(8, 6), padx=4)
        tk.Frame(parent, bg=self.BORDER, height=1).pack(fill="x", padx=4, pady=(0, 10))

    def create_widgets(self):
        # ── Header bar ────────────────────────────────────────────
        header = tk.Frame(self.root, bg=self.PANEL, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header,
                 text="  FACE RECOGNITION SYSTEM",
                 bg=self.PANEL, fg=self.TEXT,
                 font=(self.FONT_FAMILY, 13, "bold")).pack(side="left", padx=16, pady=12)
        device_txt = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        tk.Label(header,
                 text=f"Device: {device_txt}",
                 bg=self.PANEL, fg=self.SUBTEXT,
                 font=self.f_small).pack(side="right", padx=16)

        # ── Config card ───────────────────────────────────────────
        cf_outer, cf = self._card(self.root)
        cf_outer.pack(fill="x", padx=18, pady=(14, 6))
        self._section_title(cf, "Cau hinh He thong")
        grid_frame = tk.Frame(cf, bg=self.PANEL)
        grid_frame.pack(fill="x", padx=4)
        for row, (lbl, var, cmd) in enumerate([
            ("Model (.pth):",          self.selected_model_path, self.browse_model),
            ("Thu vien anh (Gallery):", self.selected_db_dir,    self.browse_db),
        ]):
            tk.Label(grid_frame, text=lbl, bg=self.PANEL, fg=self.SUBTEXT,
                     font=self.f_small).grid(row=row, column=0, sticky="w",
                                             padx=(8, 4), pady=6)
            ttk.Entry(grid_frame, textvariable=var, style="TEntry").grid(
                row=row, column=1, sticky="ew", padx=4)
            ttk.Button(grid_frame, text="Duyet...", style="Ghost.TButton",
                       command=cmd).grid(row=row, column=2, padx=(4, 10))

        cf.columnconfigure(1, weight=1)

        ttk.Button(grid_frame, text="NAP MODEL & DATABASE",
                   style="Accent.TButton",
                   command=self.load_system).grid(
            row=2, column=0, columnspan=3, pady=(8, 14), padx=10, sticky="ew")

        # ── Main area ─────────────────────────────────────────────
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill="both", expand=True, padx=18, pady=8)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # -- Cột trái: Query Image ---------------------------------
        lf_outer, lf = self._card(main)
        lf_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._section_title(lf, "Anh Can Doan")

        ttk.Button(lf, text="Chon anh...",
                   style="Ghost.TButton",
                   command=self.browse_query_image).pack(anchor="w", padx=8, pady=(0, 8))

        img_holder = tk.Frame(lf, bg="#252836", width=260, height=260)
        img_holder.pack(padx=8, pady=4)
        img_holder.pack_propagate(False)
        self.lbl_query_img = tk.Label(img_holder,
                                      text="Chua chon anh",
                                      bg="#252836", fg=self.SUBTEXT,
                                      font=self.f_small, anchor="center")
        self.lbl_query_img.pack(fill="both", expand=True)

        self.btn_predict = ttk.Button(lf, text="DOAN XEM LA AI?",
                                      style="Predict.TButton",
                                      command=self.predict_face,
                                      state="disabled")
        self.btn_predict.pack(fill="x", padx=8, pady=12)

        # -- Cột phải: Result -------------------------------------
        rf_outer, rf = self._card(main)
        rf_outer.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._section_title(rf, "Ket qua Khop nhat")

        self.lbl_result_name = tk.Label(rf,
            text="Ten nguoi: Chua co",
            bg=self.PANEL, fg=self.TEXT,
            font=self.f_result, anchor="w")
        self.lbl_result_name.pack(fill="x", padx=10, pady=(0, 4))

        self.lbl_result_score = tk.Label(rf,
            text="Do tuong dong: --",
            bg=self.PANEL, fg=self.SUBTEXT,
            font=self.f_score, anchor="w")
        self.lbl_result_score.pack(fill="x", padx=10, pady=(0, 10))

        # Progress bar độ tương đồng
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(rf, variable=self.progress_var,
                                             maximum=100, length=260)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))

        match_holder = tk.Frame(rf, bg="#252836", width=260, height=260)
        match_holder.pack(padx=8, pady=4)
        match_holder.pack_propagate(False)
        self.lbl_matched_img = tk.Label(match_holder,
                                         text="Anh doi chieu tuong ung",
                                         bg="#252836", fg=self.SUBTEXT,
                                         font=self.f_small, anchor="center")
        self.lbl_matched_img.pack(fill="both", expand=True)

        # ── Status bar ────────────────────────────────────────────
        self.status_var = tk.StringVar(value="San sang.")
        status_bar = tk.Frame(self.root, bg=self.PANEL, height=26)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self.status_var,
                 bg=self.PANEL, fg=self.SUBTEXT,
                 font=self.f_small, anchor="w").pack(side="left", padx=12)

    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pth")])
        if file_path: self.selected_model_path.set(file_path)
            
    def browse_db(self):
        dir_path = filedialog.askdirectory()
        if dir_path: self.selected_db_dir.set(dir_path)
            
    def browse_query_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.selected_query_img.set(file_path)
            img = Image.open(file_path).resize((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            self.lbl_query_img.config(image=img_tk, text="")
            self.lbl_query_img.image = img_tk
            self.status_var.set(f"Anh da chon: {os.path.basename(file_path)}")

    # ---------------- BÊ NGUYÊN XI LOGIC NẠP MODEL TỪ EVAL_ALL ----------------
    def load_system(self):
        model_path = self.selected_model_path.get()
        db_path = self.selected_db_dir.get()
        
        if not model_path or not db_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn đầy đủ file Model và thư mục Database ảnh!")
            return
            
        try:
            model_key = os.path.splitext(os.path.basename(model_path))[0]
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
            
            print(f"[GUI] Khởi tạo mô hình: {model_key} trên {self.device}")
            
            model = get_model(0, **cfg)
            
            if "clip" in model_name_lower or "dinov2" in model_name_lower or cfg.use_lora:
                from finetuning import apply_lora_model
                print("-> Attaching LoRA layers to model before loading weights...")
                apply_lora_model(
                    0, 
                    model, 
                    training_type="image_encoder_only",
                    model_name=cfg.model_name,
                    backbone_size=cfg.backbone_size, 
                    lora_target_modules=['q', 'v'],
                    lora_r=16, 
                    lora_a=16,          
                    lora_dropout=0.25, 
                    device=self.device, 
                    position="all"
                )
            
            print(f"-> Loading weights từ: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            model.backbone.load_state_dict(state_dict)
            
            self.backbone_net = model.backbone.to(self.device)
            self.backbone_net.eval()
            
            self.transform = transform_image(
                image_size=cfg.image_size, 
                normalize_type=cfg.normalize_type,
                interpolation_type=cfg.interpolation_type
            )
            
            self.gallery_features = []
            self.gallery_names = []
            self.gallery_paths = []
            
            valid_extensions = ('.jpg', '.jpeg', '.png')
            for file_name in os.listdir(db_path):
                if file_name.lower().endswith(valid_extensions):
                    full_path = os.path.join(db_path, file_name)
                    embedding = self.get_image_embedding(full_path)
                    self.gallery_features.append(embedding)
                    self.gallery_names.append(os.path.splitext(file_name)[0])
                    self.gallery_paths.append(full_path)
            
            if not self.gallery_features:
                raise ValueError("Thư mục database trống hoặc không chứa định dạng ảnh hợp lệ!")
                
            self.gallery_features = np.vstack(self.gallery_features)
            
            msg = f"He thong san sang! Da nap thanh cong {len(self.gallery_names)} nguoi dung."
            messagebox.showinfo("Thanh cong", msg)
            self.status_var.set(f"[OK] Da nap {len(self.gallery_names)} anh vao Gallery.")
            self.btn_predict.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Loi Nap He Thong", f"Khong the dong bo cau hinh voi file eval_all:\n{str(e)}")
            self.status_var.set(f"[LOI] {str(e)[:80]}")

    def get_image_embedding(self, img_path):
        """Hàm đọc ảnh và xử lý dữ liệu chuẩn qua MxNet kết hợp Flip Augmentation giống hệt eval_all.py"""
        img_mx_raw = mx.image.imdecode(open(img_path, 'rb').read())
        if img_mx_raw.shape[1] != cfg.image_size:
            img_mx_raw = mx.image.resize_short(img_mx_raw, cfg.image_size)
            
        img_np = img_mx_raw.asnumpy()
        img_transformed = self.transform(img_np)
        img_mx = mx.nd.array(img_transformed)
        
        data0 = torch.from_numpy(img_mx.asnumpy()).unsqueeze(0).to(self.device)
        img_flip = mx.ndarray.flip(data=img_mx, axis=2)
        data1 = torch.from_numpy(img_flip.asnumpy()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.backbone_net, 'encode_image'):
                out0 = self.backbone_net.encode_image(data0)
                out1 = self.backbone_net.encode_image(data1)
            else:
                out0 = self.backbone_net(data0)
                out1 = self.backbone_net(data1)
            
            if hasattr(out0, 'pooler_output'):
                out0 = out0.pooler_output
                out1 = out1.pooler_output
                
            feat0 = out0.cpu().numpy()
            feat1 = out1.cpu().numpy()
            
            import sklearn.preprocessing as preprocessing
            combined_feat = preprocessing.normalize(feat0) + preprocessing.normalize(feat1)
            combined_feat = preprocessing.normalize(combined_feat)
            
            return combined_feat

    def predict_face(self):
        query_path = self.selected_query_img.get()
        if not query_path:
            messagebox.showwarning("Cảnh báo", "Hãy chọn 1 bức ảnh để dự đoán!")
            return
            
        try:
            query_feat = self.get_image_embedding(query_path)
            similarities = np.dot(self.gallery_features, query_feat.T).flatten()
            
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            best_name = self.gallery_names[best_idx]
            best_img_path = self.gallery_paths[best_idx]
            
            percentage = max(0.0, min(100.0, (best_score + 1) / 2 * 100))
            self.progress_var.set(percentage)
            
            if best_score < 0.35:
                self.lbl_result_name.config(text="Ten nguoi: NGUOI LA (Unknown)", fg=self.DANGER)
                self.lbl_result_score.config(text=f"Do tuong dong cao nhat: {percentage:.2f}%")
                self.lbl_matched_img.config(image="", text="Khong tim thay khuon mat trong DB")
                self.status_var.set(f"Ket qua: UNKNOWN (score={best_score:.3f})")
            else:
                self.lbl_result_name.config(text=f"Ten nguoi: {best_name}", fg=self.SUCCESS)
                self.lbl_result_score.config(text=f"Do tuong dong: {percentage:.2f}%")
                self.status_var.set(f"Ket qua: {best_name}  ({percentage:.1f}%)")
                
                match_img = Image.open(best_img_path).resize((250, 250))
                match_img_tk = ImageTk.PhotoImage(match_img)
                self.lbl_matched_img.config(image=match_img_tk, text="")
                self.lbl_matched_img.image = match_img_tk
                
        except Exception as e:
            messagebox.showerror("Loi Nhan Dien", f"Xay ra loi trong qua trinh xu ly dac trung hinh anh:\n{str(e)}")
            self.status_var.set(f"[LOI] {str(e)[:80]}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()