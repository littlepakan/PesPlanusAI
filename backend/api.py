import asyncio
import gc
import io
import os
import pickle
from typing import Optional

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms
import xgboost
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ป้องกัน Error รูปภาพสูญหายหรือถูกตัดทอน
ImageFile.LOAD_TRUNCATED_IMAGES = True 

app = FastAPI(title="Pes Planus AI API")

# --- ตั้งค่า CORS ---
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

# 📌 กำหนดชื่อไฟล์โมเดลที่วางอยู่บนเซิร์ฟเวอร์ (ต้องเอาไฟล์นี้มาวางคู่กับ api.py)
MODEL_PATH = "model.pkl" 

model_lock = asyncio.Lock()
global_state = {
    "feature_extractor": None,
    "ml_model": None,
    "csv_key": None,
    "gt_map": {},
}

def parse_csv_dataframe(df: pd.DataFrame):
    df.columns = [str(c).strip().replace("\n", "").lower() for c in df.columns]
    img_col = "img_name" if "img_name" in df.columns else None
    label_col = None
    for col in ["label", "label_bin", "patient_label"]:
        if col in df.columns:
            label_col = col
            break

    gt_map = {}
    if img_col and label_col:
        for _, row in df.iterrows():
            if pd.isna(row[img_col]) or pd.isna(row[label_col]):
                continue
            rname = str(row[img_col]).strip().lower()
            b_rname = os.path.splitext(rname)[0]
            raw_lbl = str(row[label_col]).strip().lower()

            if raw_lbl in ["1", "1.0", "flatfoot", "pesplanus", "pes planus", "true"]:
                lbl = 1
            elif raw_lbl in ["0", "0.0", "normal", "false"]:
                lbl = 0
            else:
                try:
                    lbl = int(float(raw_lbl))
                except ValueError:
                    continue

            gt_map[rname] = lbl
            gt_map[b_rname] = lbl
            gt_map[f"{b_rname}.png"] = lbl
            gt_map[f"{b_rname}.jpg"] = lbl
            gt_map[f"{b_rname}.jpeg"] = lbl
    return gt_map

def apply_median_filter(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    median_img = cv2.medianBlur(img_cv, 3)
    return Image.fromarray(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))

def get_transforms():
    return transforms.Compose([
        transforms.Lambda(apply_median_filter),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class SqueezeNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        orig = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.features = orig.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)

# 🟢 Endpoint สำหรับรับภาพมาวิเคราะห์
@app.post("/api/predict")
async def predict_single_image(
    file: UploadFile = File(...),
    gt_option: str = Form("none"),
    csv_file: Optional[UploadFile] = File(None),
):
    try:
        async with model_lock:
            # 1. โหลด ML Model จากไฟล์ในเซิร์ฟเวอร์ (ถ้ายังไม่ได้โหลด)
            if global_state["ml_model"] is None:
                if not os.path.exists(MODEL_PATH):
                    raise HTTPException(status_code=500, detail=f"ไม่พบไฟล์โมเดล '{MODEL_PATH}' บนเซิร์ฟเวอร์ กรุณาตรวจสอบว่ามีไฟล์นี้อยู่คู่กับ api.py")
                try:
                    global_state["ml_model"] = joblib.load(MODEL_PATH)
                except Exception:
                    try:
                        with open(MODEL_PATH, "rb") as f:
                            global_state["ml_model"] = pickle.load(f)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"ไฟล์ Model (.pkl) เสียหาย: {str(e)}")

            # 2. โหลด Feature Extractor (ถ้ายังไม่ได้โหลด)
            if global_state["feature_extractor"] is None:
                feat_ext = SqueezeNetExtractor()
                global_state["feature_extractor"] = feat_ext.to(device).eval()

            # 3. จัดการไฟล์ CSV (ถ้าอัปโหลดมา)
            if gt_option == "upload" and csv_file:
                if global_state["csv_key"] != f"upload_{csv_file.filename}":
                    df_gt = pd.read_csv(io.BytesIO(await csv_file.read()))
                    global_state["gt_map"] = parse_csv_dataframe(df_gt)
                    global_state["csv_key"] = f"upload_{csv_file.filename}"
            else:
                global_state["gt_map"] = {}
                global_state["csv_key"] = "none"

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = get_transforms()(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = global_state["feature_extractor"](img_tensor).cpu().numpy()

        ml_model = global_state["ml_model"]

        prediction_result = int(ml_model.predict(features)[0])
        prob = float(prediction_result)

        if hasattr(ml_model, "predict_proba"):
            try:
                prob = float(ml_model.predict_proba(features)[0][1])
            except Exception as e:
                if "validate_features" in str(e) and "XGB" in type(ml_model).__name__:
                    dtest = xgboost.DMatrix(features)
                    prob_raw = ml_model.get_booster().predict(dtest)
                    prob = float(prob_raw[0])
                else:
                    prob = float(prediction_result)
        elif hasattr(ml_model, "decision_function"):
            df_val = ml_model.decision_function(features)[0]
            prob = float(1 / (1 + np.exp(-df_val)))

        fname = str(file.filename).strip().lower()
        bname = os.path.splitext(fname)[0]
        gt_label = global_state["gt_map"].get(fname) or global_state["gt_map"].get(bname)

        eval_status = "ไม่มีเฉลย"
        if gt_label is not None:
            if gt_label == 1 and prediction_result == 1:
                eval_status = "True Positive (TP)"
            elif gt_label == 0 and prediction_result == 0:
                eval_status = "True Negative (TN)"
            elif gt_label == 0 and prediction_result == 1:
                eval_status = "False Positive (FP)"
            elif gt_label == 1 and prediction_result == 0:
                eval_status = "False Negative (FN)"

        result = {
            "id": file.filename,
            "filename": file.filename,
            "prediction_class": "Pes Planus (ภาวะเท้าแบน)" if prediction_result == 1 else "Normal (ปกติ)",
            "prediction_code": prediction_result,
            "confidence": prob,
            "ground_truth": "Pes Planus (1)" if gt_label == 1 else ("Normal (0)" if gt_label == 0 else "-"),
            "eval_status": eval_status,
        }

        del img_tensor, features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")