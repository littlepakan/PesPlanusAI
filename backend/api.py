import asyncio
import gc
import io
import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ป้องกัน Error รูปภาพสูญหายหรือถูกตัดทอน
ImageFile.LOAD_TRUNCATED_IMAGES = True 

app = FastAPI(title="Pes Planus AI API (DenseNet-201)")

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
    return {"message": "Pes Planus DenseNet-201 API is running perfectly!"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 กำหนดชื่อไฟล์โมเดล DenseNet-201 ที่วางอยู่บนเซิร์ฟเวอร์
MODEL_PATH = "ens_arch_densenet201.pt" 

model_lock = asyncio.Lock()
global_state = {
    "model": None,
    "checkpoint": None,
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

def build_ft_model(name, pretrained=False):
    """ฟังก์ชันสร้างโครงสร้างโมเดล (รองรับทั้ง DenseNet และอื่นๆ เผื่ออนาคต)"""
    m = models.get_model(name, weights="DEFAULT" if pretrained else None)
    if name.startswith("densenet"):
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, 2))
    elif name.startswith("efficientnet"):
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(f"ไม่รู้จักโมเดล: {name}")
    return m

def load_single_model(filepath):
    """ฟังก์ชันโหลดโมเดลและ Metadata จากไฟล์ .pt"""
    ck = torch.load(filepath, map_location=device, weights_only=False)
    m = build_ft_model(ck["arch"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    m.to(device)
    m.eval()
    return m, ck

# 🟢 Endpoint สำหรับรับภาพมาวิเคราะห์
@app.post("/api/predict")
async def predict_single_image(
    file: UploadFile = File(...),
    gt_option: str = Form("none"),
    csv_file: Optional[UploadFile] = File(None),
):
    try:
        async with model_lock:
            # 1. โหลด Model .pt จากเซิร์ฟเวอร์ (ถ้ายังไม่ได้โหลด)
            if global_state["model"] is None:
                if not os.path.exists(MODEL_PATH):
                    raise HTTPException(
                        status_code=500, 
                        detail=f"ไม่พบไฟล์โมเดล '{MODEL_PATH}' บนเซิร์ฟเวอร์ กรุณาอัปโหลดไฟล์มาวางคู่กับ api.py"
                    )
                m, ck = load_single_model(MODEL_PATH)
                global_state["model"] = m
                global_state["checkpoint"] = ck

            # 2. จัดการไฟล์ CSV (ถ้าอัปโหลดมา)
            if gt_option == "upload" and csv_file:
                if global_state["csv_key"] != f"upload_{csv_file.filename}":
                    df_gt = pd.read_csv(io.BytesIO(await csv_file.read()))
                    global_state["gt_map"] = parse_csv_dataframe(df_gt)
                    global_state["csv_key"] = f"upload_{csv_file.filename}"
            else:
                global_state["gt_map"] = {}
                global_state["csv_key"] = "none"

        # 3. เตรียมรูปภาพ
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        m = global_state["model"]
        ck = global_state["checkpoint"]
        
        # 4. แปลงรูปภาพตาม Metadata ที่ถูกเซฟไว้ใน .pt (DenseNet = 224x224)
        tfm = transforms.Compose([
            transforms.Resize((ck["img_size"], ck["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=ck["mean"], std=ck["std"])
        ])
        
        x = tfm(image).unsqueeze(0).to(device)
        
        # 5. วิเคราะห์และทำนายผล
        with torch.no_grad():
            # ใช้ Test-Time Augmentation (TTA) พลิกภาพแนวนอนเหมือนตอนประเมินผลในโน้ตบุ๊ค
            p = (F.softmax(m(x).float(), 1) + F.softmax(m(torch.flip(x, dims=[3])).float(), 1)) / 2
            prob = float(p[0, 1].item())

        # 6. ตัดสินผลลัพธ์ด้วย Threshold ที่เซฟมา (ปกติคือ 0.5)
        thr = ck.get("threshold", 0.5)
        prediction_result = 1 if prob >= thr else 0

        # --- ตรวจสอบกับ Ground Truth (เฉลย) ---
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

        # เคลียร์หน่วยความจำ
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