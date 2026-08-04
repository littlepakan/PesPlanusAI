from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pickle
import io
import os
import pandas as pd
import xgboost 
import asyncio
import gc # 🌟 เพิ่ม import gc สำหรับจัดการคืนพื้นที่ RAM ให้เซิร์ฟเวอร์

app = FastAPI(title="Pes Planus AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Pes Planus API is running perfectly!"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🌟 โครงสร้าง SimpleNN (แก้ไขให้ไม่ต้องรับพารามิเตอร์แล้ว)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(300, 1024),          # รับ 300 ฟีเจอร์จาก RFE
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

model_lock = asyncio.Lock()
global_state = {
    "model_key": None,
    "feature_extractor": None,
    "ft_model": None,
    "nn_model": None,
    "ml_model": None,
    "rfe": None,
    "csv_key": None,
    "gt_map": {}
}

def parse_csv_dataframe(df: pd.DataFrame):
    df.columns = [str(c).strip().lower() for c in df.columns]
    img_col = next((col for col in ['img_name', 'img', 'filename', 'image_name', 'name', 'path'] if col in df.columns), None)
    label_col = next((col for col in ['label', 'imglabel', 'classification', 'class'] if col in df.columns), None)
    
    gt_map = {}
    if img_col and label_col:
        for _, row in df.iterrows():
            if pd.isna(row[img_col]) or pd.isna(row[label_col]): continue
            rname = str(row[img_col]).strip().lower()
            b_rname = os.path.splitext(rname)[0]
            raw_lbl = str(row[label_col]).strip().lower()
            
            if raw_lbl in ['1', '1.0', 'flatfoot', 'pesplanus', 'pes planus', 'true']: lbl = 1
            elif raw_lbl in ['0', '0.0', 'normal', 'false']: lbl = 0
            else:
                try: lbl = int(float(raw_lbl))
                except ValueError: continue
            
            gt_map[rname] = lbl
            gt_map[b_rname] = lbl
    return gt_map

def apply_preprocessing(img, filter_type: str):
    if filter_type == "Median Filter":
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        median_img = cv2.medianBlur(img_cv, 3)
        return Image.fromarray(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))
    elif filter_type == "Gaussian Blur":
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        blur_img = cv2.GaussianBlur(img_cv, (3, 3), 0)
        return Image.fromarray(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
    return img

def get_transforms(backbone: str, filter_type: str):
    IMG_SIZE = 224 if backbone == "GoogleNet" else 227
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_preprocessing(img, filter_type)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@app.post("/api/predict")
async def predict_single_image(
    file: UploadFile = File(...), 
    backbone: str = Form(...),
    classifier: str = Form(...),
    filter_type: str = Form("Median Filter"),
    gt_option: str = Form("none"),
    weight_file: Optional[UploadFile] = File(None),
    clf_file: Optional[UploadFile] = File(None),
    rfe_file: Optional[UploadFile] = File(None),
    csv_file: Optional[UploadFile] = File(None)
):
    try:
        async with model_lock:
            # --- 1. โหมด Fine-Tuning (.pth ตัวเดียว) ---
            if classifier == "Fine-Tuning":
                if not weight_file:
                    raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์ Weights (.pth) สำหรับ Fine-Tuning")
                
                current_key = f"FT_{backbone}_{weight_file.filename}"
                if global_state["model_key"] != current_key:
                    w_bytes = await weight_file.read()
                    if backbone == "GoogleNet":
                        model = models.googlenet(aux_logits=False)
                        model.fc = nn.Linear(1024, 2)
                    else:
                        model = models.squeezenet1_1()
                        model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(512, 2, kernel_size=(1, 1)), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
                    
                    # 🌟 ใส่ try-except ดักจับ Mismatch Error (Human Error)
                    try:
                        model.load_state_dict(torch.load(io.BytesIO(w_bytes), map_location=device, weights_only=False))
                    except Exception as e:
                        err_str = str(e)
                        if "Missing key(s)" in err_str or "Unexpected key(s)" in err_str or "size mismatch" in err_str:
                            raise HTTPException(
                                status_code=400, 
                                detail=f"ไฟล์ Weights (.pth) ไม่ตรงกับ Backbone ที่เลือก! คุณเลือก [{backbone}] แต่ไฟล์ที่อัปโหลดไม่ใช่โครงสร้างของโมเดลนี้"
                            )
                        raise HTTPException(status_code=400, detail=f"ไม่สามารถโหลดไฟล์โมเดลได้: {err_str}")

                    global_state["ft_model"] = model.to(device).eval()
                    global_state["model_key"] = current_key

            # --- 2. โหมด Neural Network (ใช้ไฟล์ .pth + RFE Selector .pkl) ---
            elif classifier == "NeuralNetwork":
                if not weight_file or not rfe_file:
                    raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์ Weights (.pth) และ RFE Selector (.pkl) ให้ครบถ้วน")
                
                current_key = f"NN_{backbone}_{weight_file.filename}_{rfe_file.filename}"
                if global_state["model_key"] != current_key:
                    w_bytes = await weight_file.read()
                    rfe_bytes = await rfe_file.read()
                    
                    global_state["rfe"] = pickle.loads(rfe_bytes)

                    if backbone == "GoogleNet":
                        feat_ext = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                        feat_ext.fc = nn.Identity()
                    else:
                        class SqueezeNetExtractor(nn.Module):
                            def __init__(self):
                                super().__init__()
                                orig = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
                                self.features = orig.features
                                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                            def forward(self, x):
                                return torch.flatten(self.pool(self.features(x)), 1)
                        feat_ext = SqueezeNetExtractor()
                    
                    global_state["feature_extractor"] = feat_ext.to(device).eval()

                    nn_model = SimpleNN()
                    
                    # 🌟 ใส่ try-except ดักจับ Mismatch Error ของ Neural Network (Human Error)
                    try:
                        nn_model.load_state_dict(torch.load(io.BytesIO(w_bytes), map_location=device, weights_only=False))
                    except Exception as e:
                        err_str = str(e)
                        if "Missing key(s)" in err_str or "Unexpected key(s)" in err_str or "size mismatch" in err_str:
                            raise HTTPException(
                                status_code=400, 
                                detail=f"ไฟล์ Weights (.pth) ไม่ตรงกับ Classifier ที่เลือก! คุณเลือก [{backbone}] แต่ไฟล์ .pth ที่อัปโหลดไม่รองรับ"
                            )
                        raise HTTPException(status_code=400, detail=f"ไม่สามารถโหลดไฟล์โมเดลได้: {err_str}")

                    global_state["nn_model"] = nn_model.to(device).eval()
                    global_state["model_key"] = current_key

            # --- 3. โหมด Machine Learning (SVM, XGBoost, RF, DT ใช้ .pkl + RFE) ---
            else:
                if not clf_file or not rfe_file:
                    raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์ Classifier (.pkl) และ RFE Selector (.pkl)")
                
                current_key = f"ML_{backbone}_{classifier}_{clf_file.filename}_{rfe_file.filename}"
                if global_state["model_key"] != current_key:
                    global_state["ml_model"] = pickle.loads(await clf_file.read())
                    global_state["rfe"] = pickle.loads(await rfe_file.read())

                    if backbone == "GoogleNet":
                        feat_ext = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                        feat_ext.fc = nn.Identity()
                    else:
                        class SqueezeNetExtractor(nn.Module):
                            def __init__(self):
                                super().__init__()
                                orig = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
                                self.features = orig.features
                                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                            def forward(self, x):
                                return torch.flatten(self.pool(self.features(x)), 1)
                        feat_ext = SqueezeNetExtractor()
                    
                    global_state["feature_extractor"] = feat_ext.to(device).eval()
                    global_state["model_key"] = current_key

            # --- 4. จัดการ CSV เฉลย ---
            if gt_option == "upload" and csv_file:
                if global_state["csv_key"] != f"upload_{csv_file.filename}":
                    df_gt = pd.read_csv(io.BytesIO(await csv_file.read()))
                    global_state["gt_map"] = parse_csv_dataframe(df_gt)
                    global_state["csv_key"] = f"upload_{csv_file.filename}"
            else: 
                global_state["gt_map"] = {}
                global_state["csv_key"] = "none"

        # --- ทำนายผล ---
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = get_transforms(backbone, filter_type)(image).unsqueeze(0).to(device)

        if classifier == "Fine-Tuning":
            with torch.no_grad():
                outputs = global_state["ft_model"](img_tensor)
                _, preds = torch.max(outputs.data, 1)
                prob = torch.softmax(outputs, dim=1)[:, 1].item()
                prediction_result = preds.item()
                
        elif classifier == "NeuralNetwork":
            with torch.no_grad():
                features = global_state["feature_extractor"](img_tensor).cpu().numpy()
            features_opt = global_state["rfe"].transform(features)
            
            with torch.no_grad():
                feats_tensor = torch.FloatTensor(features_opt).to(device)
                outputs = global_state["nn_model"](feats_tensor)
                _, preds = torch.max(outputs.data, 1)
                prob = torch.softmax(outputs, dim=1)[:, 1].item()
                prediction_result = preds.item()
                
        else:
            with torch.no_grad():
                features = global_state["feature_extractor"](img_tensor).cpu().numpy()
            features_opt = global_state["rfe"].transform(features)
            
            prediction_result = int(global_state["ml_model"].predict(features_opt)[0])
            
            # 🌟 การจัดการ Probability แบบปลอดภัย (แก้บั๊ก XGBoost validate_features)
            prob = float(prediction_result)
            if hasattr(global_state["ml_model"], "predict_proba"):
                try:
                    prob = float(global_state["ml_model"].predict_proba(features_opt)[0][1])
                except TypeError as e:
                    if "validate_features" in str(e) and classifier == "XGBoost":
                        dtest = xgboost.DMatrix(features_opt)
                        prob_raw = global_state["ml_model"].get_booster().predict(dtest)
                        prob = float(prob_raw[0])
                    else:
                        prob = float(prediction_result)

        # เช็กกับ Ground Truth
        fname = str(file.filename).strip().lower()
        bname = os.path.splitext(fname)[0]
        gt_label = global_state["gt_map"].get(fname) or global_state["gt_map"].get(bname)
        
        eval_status = "ไม่มีเฉลย"
        if gt_label is not None:
            if gt_label == 1 and prediction_result == 1: eval_status = "True Positive (TP)"
            elif gt_label == 0 and prediction_result == 0: eval_status = "True Negative (TN)"
            elif gt_label == 0 and prediction_result == 1: eval_status = "False Positive (FP)"
            elif gt_label == 1 and prediction_result == 0: eval_status = "False Negative (FN)"

        # 🌟 สั่งคืนพื้นที่ RAM ทันทีหลังทำนายเสร็จ
        if 'img_tensor' in locals(): del img_tensor
        if 'features' in locals(): del features
        if 'features_opt' in locals(): del features_opt
        if 'outputs' in locals(): del outputs
        
        gc.collect() # ล้างขยะออกจาก Memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "id": file.filename, 
            "filename": file.filename,
            "prediction_class": "Pes Planus (1)" if prediction_result == 1 else "Normal (0)",
            "prediction_code": prediction_result,
            "confidence": f"{prob:.4f}",
            "ground_truth": "Pes Planus (1)" if gt_label == 1 else ("Normal (0)" if gt_label == 0 else "-"),
            "eval_status": eval_status
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))