import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import numpy as np
import mxnet as mx

# Ensure system path includes current working directory
sys.path.append(os.path.join(os.getcwd()))

from config.config_eval import config as cfg
from backbone import get_model
from data.transform import transform_image

class FaceRecognitionGUI:
    # ── Colors & Fonts ──────────────────────────────────────────────
    BG          = "#0F1117"   # Dark background
    PANEL       = "#1A1D27"   # Card / Panel background
    BORDER      = "#2A2D3E"   # Border color
    ACCENT      = "#4F8EF7"   # Main Blue accent
    ACCENT2     = "#7C5CFC"   # Purple accent for actions
    SUCCESS     = "#2ECC71"
    DANGER      = "#E74C3C"
    TEXT        = "#E8EAF0"
    SUBTEXT     = "#8890A8"
    FONT_FAMILY = "Roboto"    # Standard robust font for Linux X11/Windows

    # ALL 18 MODELS FROM THE PAPER
    MODELS_MAPPING = {
        # --- 1. BASELINE MODELS ---
        "Baseline_Small_CASIA-WebFace": "weights/Baseline/Small/Baseline_Small_CASIA-WebFace.pth",
        "Baseline_Small_MS1MV2": "weights/Baseline/Small/Baseline_Small_MS1MV2.pth",
        "Baseline_Small_WebFace4M": "weights/Baseline/Small/Baseline_Small_WebFace4M.pth",
        "Baseline_Large_CASIA-WebFace": "weights/Baseline/Large/Baseline_Large_CASIA-WebFace.pth",
        "Baseline_Large_MS1MV2": "weights/Baseline/Large/Baseline_Large_MS1MV2.pth",
        "Baseline_Large_WebFace4M": "weights/Baseline/Large/Baseline_Large_WebFace4M.pth",

        # --- 2. CLIP BASED MODELS ---
        "CLIP_Base_CASIA-WebFace": "weights/CLIP/Base/CLIP_Base_CASIA-WebFace.pth",
        "CLIP_Base_MS1MV2": "weights/CLIP/Base/CLIP_Base_MS1MV2.pth",
        "CLIP_Base_WebFace4M": "weights/CLIP/Base/CLIP_Base_WebFace4M.pth",
        "CLIP_Large_CASIA-WebFace": "weights/CLIP/Large/CLIP_Large_CASIA-WebFace.pth",
        "CLIP_Large_MS1MV2": "weights/CLIP/Large/CLIP_Large_MS1MV2.pth",
        "CLIP_Large_WebFace4M": "weights/CLIP/Large/CLIP_Large_WebFace4M.pth",

        # --- 3. DINOv2 BASED MODELS ---
        "DINOv2_Small_CASIA-WebFace": "weights/DINOv2/Small/DINOv2_Small_CASIA-WebFace.pth",
        "DINOv2_Small_MS1MV2": "weights/DINOv2/Small/DINOv2_Small_MS1MV2.pth",
        "DINOv2_Small_WebFace4M": "weights/DINOv2/Small/DINOv2_Small_WebFace4M.pth",
        "DINOv2_Base_CASIA-WebFace": "weights/DINOv2/Base/DINOv2_Base_CASIA-WebFace.pth",
        "DINOv2_Base_MS1MV2": "weights/DINOv2/Base/DINOv2_Base_MS1MV2.pth",
        "DINOv2_Base_WebFace4M": "weights/DINOv2/Base/DINOv2_Base_WebFace4M.pth",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1050x700")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)

        # UI Variables
        self.selected_model_key = tk.StringVar()
        self.selected_db_dir = tk.StringVar()
        self.selected_query_img = tk.StringVar()

        # Hardware setup
        self.backbone_net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_fonts()
        self._setup_styles()
        self.create_widgets()

    def _setup_fonts(self):
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

        # Entry & Combobox
        style.configure("TEntry",
            fieldbackground="#252836", foreground=TEXT,
            insertcolor=TEXT, borderwidth=1, relief="flat")
        style.map("TEntry", bordercolor=[("focus", ACCENT), ("!focus", BORDER)])

        style.configure("TCombobox",
            fieldbackground="#252836", foreground=TEXT,
            background=PANEL, borderwidth=1, relief="flat")
        style.map("TCombobox", 
            fieldbackground=[("readonly", "#252836")],
            foreground=[("readonly", TEXT)])

        # Main Load button
        style.configure("Accent.TButton",
            background=ACCENT, foreground="#FFFFFF",
            font=self.f_btn, borderwidth=0, padding=(14, 8))
        style.map("Accent.TButton",
            background=[("active", "#3A7BE8"), ("disabled", "#2A2D3E")],
            foreground=[("disabled", SUBTEXT)])

        # Secondary Browse button
        style.configure("Ghost.TButton",
            background=PANEL, foreground=ACCENT,
            font=self.f_label, borderwidth=1, relief="flat", padding=(10, 6))
        style.map("Ghost.TButton",
            background=[("active", BORDER)])

        # Inference button
        style.configure("Predict.TButton",
            background=self.ACCENT2, foreground="#FFFFFF",
            font=(self.FONT_FAMILY, 11, "bold"), borderwidth=0, padding=(14, 10))
        style.map("Predict.TButton",
            background=[("active", "#6A4DE8"), ("disabled", "#2A2D3E")],
            foreground=[("disabled", SUBTEXT)])

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
        self._section_title(cf, "Configuration")
        grid_frame = tk.Frame(cf, bg=self.PANEL)
        grid_frame.pack(fill="x", padx=4)
        
        # Row 0: Combobox for selecting models
        tk.Label(grid_frame, text="Select Model:", bg=self.PANEL, fg=self.SUBTEXT,
                 font=self.f_small).grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
        
        self.model_combo = ttk.Combobox(grid_frame, textvariable=self.selected_model_key, 
                                        values=list(self.MODELS_MAPPING.keys()), state="readonly", style="TCombobox")
        self.model_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=4)
        if list(self.MODELS_MAPPING.keys()):
            self.model_combo.current(0)

        # Row 1: Gallery Directory Selection
        tk.Label(grid_frame, text="Gallery Directory:", bg=self.PANEL, fg=self.SUBTEXT,
                 font=self.f_small).grid(row=1, column=0, sticky="w", padx=(8, 4), pady=6)
        ttk.Entry(grid_frame, textvariable=self.selected_db_dir, style="TEntry").grid(
            row=1, column=1, sticky="ew", padx=4)
        ttk.Button(grid_frame, text="Browse...", style="Ghost.TButton",
                   command=self.browse_db).grid(row=1, column=2, padx=(4, 10))

        grid_frame.columnconfigure(1, weight=1)

        ttk.Button(grid_frame, text="INITIALIZE MODEL & GALLERY",
                   style="Accent.TButton",
                   command=self.load_system).grid(
            row=2, column=0, columnspan=3, pady=(8, 14), padx=10, sticky="ew")

        # ── Main area ─────────────────────────────────────────────
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill="both", expand=True, padx=18, pady=8)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # -- Left Column: Query Image ---------------------------------
        lf_outer, lf = self._card(main)
        lf_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._section_title(lf, "Query Input")

        ttk.Button(lf, text="Select Image...",
                   style="Ghost.TButton",
                   command=self.browse_query_image).pack(anchor="w", padx=8, pady=(0, 8))

        img_holder = tk.Frame(lf, bg="#252836", width=260, height=260)
        img_holder.pack(padx=8, pady=4)
        img_holder.pack_propagate(False)
        self.lbl_query_img = tk.Label(img_holder,
                                      text="No Image Selected",
                                      bg="#252836", fg=self.SUBTEXT,
                                      font=self.f_small, anchor="center")
        self.lbl_query_img.pack(fill="both", expand=True)

        self.btn_predict = ttk.Button(lf, text="Run Inference",
                                      style="Predict.TButton",
                                      command=self.predict_face,
                                      state="disabled")
        self.btn_predict.pack(fill="x", padx=8, pady=12)

        # -- Right Column: Match Result -------------------------------------
        rf_outer, rf = self._card(main)
        rf_outer.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self._section_title(rf, "Match Result")

        self.lbl_result_name = tk.Label(rf,
            text="Name: Unknown",
            bg=self.PANEL, fg=self.TEXT,
            font=self.f_result, anchor="w")
        self.lbl_result_name.pack(fill="x", padx=10, pady=(0, 4))

        self.lbl_result_score = tk.Label(rf,
            text="Similarity Score: --",
            bg=self.PANEL, fg=self.SUBTEXT,
            font=self.f_score, anchor="w")
        self.lbl_result_score.pack(fill="x", padx=10, pady=(0, 10))

        # Similarity progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(rf, variable=self.progress_var,
                                             maximum=100, length=260)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))

        match_holder = tk.Frame(rf, bg="#252836", width=260, height=260)
        match_holder.pack(padx=8, pady=4)
        match_holder.pack_propagate(False)
        self.lbl_matched_img = tk.Label(match_holder,
                                         text="No Match Target Available",
                                         bg="#252836", fg=self.SUBTEXT,
                                         font=self.f_small, anchor="center")
        self.lbl_matched_img.pack(fill="both", expand=True)

        # ── Status bar ────────────────────────────────────────────
        self.status_var = tk.StringVar(value="System Ready.")
        status_bar = tk.Frame(self.root, bg=self.PANEL, height=26)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self.status_var,
                 bg=self.PANEL, fg=self.SUBTEXT,
                 font=self.f_small, anchor="w").pack(side="left", padx=12)
            
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
            self.status_var.set(f"Selected Query Image: {os.path.basename(file_path)}")

    # ---------------- BACKBONE LOAD LOGIC ALIGNED WITH EVAL_ALL ----------------
    def load_system(self):
        model_key = self.selected_model_key.get()
        db_path = self.selected_db_dir.get()
        
        if not model_key or not db_path:
            messagebox.showerror("Error", "Please pick a model configuration and gallery directory!")
            return

        model_path = self.MODELS_MAPPING.get(model_key)
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Weight file not found at:\n{model_path}\nPlease verify your weights directory structure.")
            return
            
        try:
            model_name_lower = model_key.lower()
            
            # Setup architectures properties
            if "clip" in model_name_lower:
                cfg.model_name = "clip"
                cfg.backbone_size = "ViT-B/16" if "base" in model_name_lower else "ViT-L/14"
                cfg.image_size = 224
            elif "dinov2" in model_name_lower:
                cfg.model_name = "dinov2"
                cfg.backbone_size = "small" if "small" in model_name_lower else "base"
                cfg.image_size = 224
            else:
                cfg.model_name = "baseline"
                cfg.backbone_size = "small" if "small" in model_name_lower else "large"
                cfg.image_size = 112
            
            print(f"[GUI] Initializing framework architecture: {model_key} on {self.device}")
            
            model = get_model(0, **cfg)
            
            if "clip" in model_name_lower or "dinov2" in model_name_lower or cfg.use_lora:
                from finetuning import apply_lora_model
                print("-> Attaching LoRA layers to model architecture before loading weights...")
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
            
            print(f"-> Loading model state dictionary from: {model_path}")
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
                raise ValueError("Gallery directory contains no valid image configurations format!")
                
            self.gallery_features = np.vstack(self.gallery_features)
            
            msg = f"System Ready! Successfully loaded {len(self.gallery_names)} profiles with {model_key} backend."
            messagebox.showinfo("Success", msg)
            self.status_var.set(f"[OK] Parsed {len(self.gallery_names)} identities into Gallery database.")
            self.btn_predict.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Initialization Failure", f"Failed to align setup configuration:\n{str(e)}")
            print(str(e))
            self.status_var.set(f"[ERR] {str(e)[:80]}")

    def get_image_embedding(self, img_path):
        img_mx_raw = mx.image.imdecode(open(img_path, 'rb').read())
        model_key = self.selected_model_key.get().lower()
        
        # ── TÁCH BIỆT LOGIC XỬ LÝ ───────────────────────────────────
        if "clip" in model_key or "dinov2" in model_key:
            # 1. Với dòng Vision Transformer: Bắt buộc ép ảnh vuông tuyệt đối 224x224
            new_w = cfg.image_size
            new_h = cfg.image_size
            img_mx_raw = mx.image.imresize(img_mx_raw, new_w, new_h)
        else:
            # 2. Với dòng Baseline CNN: Giữ nguyên logic tính tỉ lệ cạnh ngắn nhất và làm tròn patch hệ số 14
            h, w, _ = img_mx_raw.shape
            if h < w:
                new_h = cfg.image_size
                new_w = int(w * (cfg.image_size / h))
            else:
                new_w = cfg.image_size
                new_h = int(h * (cfg.image_size / w))
                
            new_w = int(round(new_w / 14) * 14)
            new_h = int(round(new_h / 14) * 14)
            new_w = max(14, new_w)
            new_h = max(14, new_h)
            img_mx_raw = mx.image.imresize(img_mx_raw, new_w, new_h)
        # ────────────────────────────────────────────────────────────
            
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
            messagebox.showwarning("Warning", "Please choose a query image to compute inference target!")
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
                self.lbl_result_name.config(text="Name: UNKNOWN PROFILE", fg=self.DANGER)
                self.lbl_result_score.config(text=f"Highest Similarity : {percentage:.2f}%")
                self.lbl_matched_img.config(image="", text="No match found above baseline threshold")
                self.status_var.set(f"Result: UNKNOWN (score={best_score:.3f})")
            else:
                self.lbl_result_name.config(text=f"Name: {best_name}", fg=self.SUCCESS)
                self.lbl_result_score.config(text=f"Similarity : {percentage:.2f}%")
                self.status_var.set(f"Result: {best_name} ({percentage:.1f}%)")
                
                match_img = Image.open(best_img_path).resize((250, 250))
                match_img_tk = ImageTk.PhotoImage(match_img)
                self.lbl_matched_img.config(image=match_img_tk, text="")
                self.lbl_matched_img.image = match_img_tk
                
        except Exception as e:
            messagebox.showerror("Inference Failure", f"Error during tensor embedding evaluation phase:\n{str(e)}")
            self.status_var.set(f"[ERR] {str(e)[:80]}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()