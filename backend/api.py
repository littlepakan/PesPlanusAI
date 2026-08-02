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

app = FastAPI(title="Pes Planus AI API")
@app.get("/")
def read_root():
    return {"message": "Pes Planus API is running perfectly! (FastAPI default CORS enabled)"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pes-planus-ai.vercel.app",  # อนุญาตให้เว็บ Vercel ของคุณเข้าถึงได้
        "http://localhost:3000",             # เผื่อไว้ทดสอบในเครื่องตัวเอง
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "Pes Planus API is running perfectly! (Middleware CORS enabled)"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ตัวแปร Global สำหรับเก็บ Caching เพื่อความรวดเร็ว
model_lock = asyncio.Lock()
global_state = {
    "model_name": None,
    "model": None,
    "ml_model": None,
    "rfe": None,
    "csv_key": None,
    "gt_map": {}
}

# 🌟 ฟังก์ชันกรองเฉพาะคอลัมน์ที่จำเป็นใน CSV (ประหยัด RAM)
def valid_csv_cols(col_name):
    valid_names = ['img_name', 'img', 'filename', 'image_name', 'name', 'label']
    return col_name in valid_names

# ฟังก์ชันดึงค่าจาก DataFrame มาทำ Mapping
def parse_csv_dataframe(df: pd.DataFrame):
    img_col = next((col for col in ['img_name', 'img', 'filename', 'image_name', 'name'] if col in df.columns), None)
    gt_map = {}
    if img_col and 'label' in df.columns:
        for _, row in df.iterrows():
            rname = str(row[img_col]).strip()
            b_rname = os.path.splitext(rname)[0]
            lbl = int(row['label'])
            gt_map[rname] = lbl
            gt_map[b_rname] = lbl
    return gt_map

# ฟังก์ชันจัดการ Preprocessing (Filter)
def apply_preprocessing(img, filter_type: str):
    if filter_type == "Median Filter":
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        median_img = cv2.medianBlur(img_cv, 3)
        final_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_img)
    elif filter_type == "Gaussian Blur":
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        blur_img = cv2.GaussianBlur(img_cv, (3, 3), 0)
        final_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_img)
    return img

def get_transforms(model_name: str, filter_type: str):
    IMG_SIZE = 227 if model_name == "SqueezeNet" else 224
    return transforms.Compose([
        transforms.Lambda(lambda img: apply_preprocessing(img, filter_type)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@app.post("/api/predict")
async def predict_single_image(
    file: UploadFile = File(...), # 🌟 เปลี่ยนกลับมารับไฟล์เดียว เพื่อให้รองรับการกดยกเลิกแบบ Real-time
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
        # 1. จัดการโมเดลและไฟล์ CSV ภายใต้ Lock (กันพังเวลายิงรัวๆ พร้อมกัน)
        async with model_lock:
            # --- โหลดโมเดล ---
            if classifier == "Fine-Tuning":
                if not weight_file: raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์ Weights (.pth)")
                
                if global_state["model_name"] != weight_file.filename:
                    if backbone == "SqueezeNet":
                        model = models.squeezenet1_1()
                        model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(512, 2, kernel_size=(1, 1)), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
                    else:
                        model = models.googlenet(aux_logits=False)
                        model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(1024, 2))
                    
                    weight_bytes = await weight_file.read()
                    model.load_state_dict(torch.load(io.BytesIO(weight_bytes), map_location=device))
                    model.to(device)
                    model.eval()
                    
                    global_state["model"] = model
                    global_state["model_name"] = weight_file.filename
            else:
                if not clf_file or not rfe_file: raise HTTPException(status_code=400, detail="อัปโหลด .pkl ไม่ครบ")
                
                if global_state["model_name"] != clf_file.filename:
                    clf_bytes = await clf_file.read()
                    rfe_bytes = await rfe_file.read()
                    global_state["ml_model"] = pickle.loads(clf_bytes)
                    global_state["rfe"] = pickle.loads(rfe_bytes)

                    if backbone == "SqueezeNet":
                        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
                        model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
                    else:
                        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                        model.aux_logits = False
                        model.fc = nn.Identity()
                    
                    model.to(device)
                    model.eval()
                    global_state["model"] = model
                    global_state["model_name"] = clf_file.filename

            # --- อ่านไฟล์เฉลย CSV (ใช้ usecols) ---
            if gt_option == "upload" and csv_file:
                if global_state["csv_key"] != f"upload_{csv_file.filename}":
                    try:
                        csv_bytes = await csv_file.read()
                        df_gt = pd.read_csv(io.BytesIO(csv_bytes), usecols=valid_csv_cols)
                        global_state["gt_map"] = parse_csv_dataframe(df_gt)
                        global_state["csv_key"] = f"upload_{csv_file.filename}"
                    except Exception as e:
                        print(f"CSV Parse Error: {e}")
                        raise HTTPException(status_code=500, detail=f"อ่านไฟล์ CSV ที่อัปโหลดไม่สำเร็จ: {str(e)}")
            elif gt_option == "default":
                if global_state["csv_key"] != "default":
                    # แก้ไขให้อ้างอิงพาธจากตำแหน่งไฟล์ api.py
                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                    csv_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".csv")]
                    
                    if csv_files:
                        try:
                            csv_path = os.path.join(BASE_DIR, csv_files[0])
                            df_gt = pd.read_csv(csv_path, usecols=valid_csv_cols)
                            global_state["gt_map"] = parse_csv_dataframe(df_gt)
                            global_state["csv_key"] = "default"
                        except Exception as e:
                            print(f"Default CSV Error: {e}")
                            raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {str(e)}")
                    else:
                        global_state["gt_map"] = {}
                        global_state["csv_key"] = "none"
                        # แจ้งเตือน Error ชัดเจนหากไม่พบไฟล์ .csv บน Production (Render)
                        raise HTTPException(
                            status_code=404, 
                            detail="ไม่พบไฟล์เฉลย (.csv) สำหรับค่า Default บนเซิร์ฟเวอร์ กรุณาอัปโหลดไฟล์ขึ้นไปยัง Render"
                        )
            else: # none
                global_state["gt_map"] = {}
                global_state["csv_key"] = "none"

        # 2. วิเคราะห์ภาพ
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        eval_transforms = get_transforms(backbone, filter_type)
        img_tensor = eval_transforms(image).unsqueeze(0).to(device)

        prediction_result = -1
        prob = 0.0

        if classifier == "Fine-Tuning":
            with torch.no_grad():
                outputs = global_state["model"](img_tensor)
                _, preds = torch.max(outputs.data, 1)
                prediction_result = preds.item()
                probs = torch.softmax(outputs, dim=1)
                prob = probs[:, 1].item()
        else:
            with torch.no_grad():
                features = global_state["model"](img_tensor).cpu().numpy()
            features_opt = global_state["rfe"].transform(features)
            prediction_result = int(global_state["ml_model"].predict(features_opt)[0])
            if hasattr(global_state["ml_model"], "predict_proba"):
                prob = float(global_state["ml_model"].predict_proba(features_opt)[0][1])
            else:
                prob = float(prediction_result)

        # 3. เช็กความถูกต้องกับเฉลย
        fname = file.filename
        bname = os.path.splitext(fname)[0]
        gt_label = None
        eval_status = "ไม่มีเฉลย"

        if global_state["gt_map"]:
            if fname in global_state["gt_map"]: gt_label = global_state["gt_map"][fname]
            elif bname in global_state["gt_map"]: gt_label = global_state["gt_map"][bname]
            
            if gt_label is not None:
                if gt_label == 1 and prediction_result == 1: eval_status = "True Positive (TP)"
                elif gt_label == 0 and prediction_result == 0: eval_status = "True Negative (TN)"
                elif gt_label == 0 and prediction_result == 1: eval_status = "False Positive (FP)"
                elif gt_label == 1 and prediction_result == 0: eval_status = "False Negative (FN)"

        # ส่งผลลัพธ์กลับ
        return {
            "id": file.filename, 
            "filename": fname,
            "prediction_class": "Pes Planus (1)" if prediction_result == 1 else "Normal (0)",
            "prediction_code": prediction_result,
            "confidence": f"{prob:.4f}",
            "ground_truth": "Pes Planus (1)" if gt_label == 1 else ("Normal (0)" if gt_label == 0 else "-"),
            "eval_status": eval_status
        }

    except HTTPException:
        # เพื่อให้ส่ง HTTP Error แบบเจาะจงที่เราสร้างไว้กลับไปได้เลย
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))