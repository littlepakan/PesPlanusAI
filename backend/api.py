import asyncio
import gc
import io
import os
import pickle
import tempfile  # <-- เพิ่ม import tempfile
from typing import List, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFile  # <-- เพิ่ม ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms
import xgboost
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# --- ตั้งค่าการอ่านรูปภาพ ---
# ป้องกัน Error รูปภาพสูญหายหรือถูกตัดทอนระหว่างการส่งผ่าน HTTP FormData
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

model_lock = asyncio.Lock()
global_state = {
    "model_key": None,
    "feature_extractor": None,
    "ft_model": None,
    "ml_model": None,
    "rfe": None,
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@app.post("/api/predict")
async def predict_batch_images(
    files: List[UploadFile] = File(...),
    backbone: str = Form(...),
    classifier: str = Form(...),
    filter_type: str = Form("Median Filter"),
    gt_option: str = Form("none"),
    weight_file: Optional[UploadFile] = File(None),  # สำหรับ Fine-Tuning (.pth)
    clf_file: Optional[UploadFile] = File(None),     # สำหรับ ML Models (.pkl)
    rfe_file: Optional[UploadFile] = File(None),     # สำหรับ ML Models (.pkl)
    csv_file: Optional[UploadFile] = File(None),
):
    try:
        async with model_lock:
            clf_target = classifier.strip().lower().replace(" ", "")

            # ------------------------------------------------------------------
            # โหมดที่ 1: Fine-Tuning (.pth ไฟล์เดียว)
            # ------------------------------------------------------------------
            if clf_target in ["fine-tuning", "finetuning", "ft"]:
                if not weight_file:
                    raise HTTPException(
                        status_code=400,
                        detail="กรุณาอัปโหลดไฟล์ Weights (.pth) สำหรับ Fine-Tuning"
                    )

                current_key = f"FT_{backbone}_{weight_file.filename}"
                if global_state["model_key"] != current_key:
                    w_bytes = await weight_file.read()
                    
                    if backbone == "GoogleNet":
                        model = models.googlenet(aux_logits=False)
                        model.fc = nn.Linear(1024, 2)
                    else:
                        model = models.squeezenet1_1()
                        model.classifier = nn.Sequential(
                            nn.Dropout(p=0.5),
                            nn.Conv2d(512, 2, kernel_size=(1, 1)),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten()
                        )

                    try:
                        model.load_state_dict(
                            torch.load(io.BytesIO(w_bytes), map_location=device, weights_only=False)
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"โหลดไฟล์ .pth ไม่สำเร็จ อาจไม่ตรงกับ Backbone [{backbone}]: {str(e)}"
                        )

                    global_state["ft_model"] = model.to(device).eval()
                    global_state["model_key"] = current_key

            # ------------------------------------------------------------------
            # โหมดที่ 2: ML Models / Classical Classifiers (.pkl 2 ไฟล์)
            # ------------------------------------------------------------------
            else:
                if not clf_file or not rfe_file:
                    raise HTTPException(
                        status_code=400,
                        detail="กรุณาอัปโหลดไฟล์ Classifier / Model (.pkl) และ RFE Selector (.pkl) ให้ครบถ้วน"
                    )

                current_key = f"ML_{backbone}_{clf_target}_{clf_file.filename}_{rfe_file.filename}"
                if global_state["model_key"] != current_key:
                    clf_bytes = await clf_file.read()
                    rfe_bytes = await rfe_file.read()

                    clf_tmp_path = None
                    rfe_tmp_path = None

                    try:
                        # สร้างไฟล์ชั่วคราวเพื่อให้ Joblib อ่านจาก File System โดยตรง (แก้ปัญหา stream corrupted)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f_clf:
                            f_clf.write(clf_bytes)
                            clf_tmp_path = f_clf.name
                            
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f_rfe:
                            f_rfe.write(rfe_bytes)
                            rfe_tmp_path = f_rfe.name

                        # โหลด Classifier (.pkl)
                        try:
                            global_state["ml_model"] = joblib.load(clf_tmp_path)
                        except Exception:
                            try:
                                global_state["ml_model"] = pickle.loads(clf_bytes)
                            except Exception as e:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"ไฟล์โมเดล (.pkl) เสียหายหรือไม่สามารถแกะไฟล์ได้: {str(e)}"
                                )

                        # โหลด RFE Selector (.pkl)
                        try:
                            global_state["rfe"] = joblib.load(rfe_tmp_path)
                        except Exception:
                            try:
                                global_state["rfe"] = pickle.loads(rfe_bytes)
                            except Exception as e:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"ไฟล์ RFE Selector (.pkl) เสียหาย: {str(e)}"
                                )
                                
                    finally:
                        # ลบไฟล์ชั่วคราวเพื่อคืนพื้นที่
                        if clf_tmp_path and os.path.exists(clf_tmp_path):
                            os.remove(clf_tmp_path)
                        if rfe_tmp_path and os.path.exists(rfe_tmp_path):
                            os.remove(rfe_tmp_path)

                    # โหลด Feature Extractor ตาม Backbone
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

            # --- จัดการ CSV Ground Truth ---
            if gt_option == "upload" and csv_file:
                if global_state["csv_key"] != f"upload_{csv_file.filename}":
                    df_gt = pd.read_csv(io.BytesIO(await csv_file.read()))
                    global_state["gt_map"] = parse_csv_dataframe(df_gt)
                    global_state["csv_key"] = f"upload_{csv_file.filename}"
            else:
                global_state["gt_map"] = {}
                global_state["csv_key"] = "none"

        # ======================================================================
        # การทำนายผล (Prediction Loop)
        # ======================================================================
        results = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            img_tensor = get_transforms(backbone, filter_type)(image).unsqueeze(0).to(device)

            if clf_target in ["fine-tuning", "finetuning", "ft"]:
                with torch.no_grad():
                    outputs = global_state["ft_model"](img_tensor)
                    _, preds = torch.max(outputs.data, 1)
                    prob = torch.softmax(outputs, dim=1)[:, 1].item()
                    prediction_result = preds.item()
            else:
                with torch.no_grad():
                    features = global_state["feature_extractor"](img_tensor).cpu().numpy()

                features_opt = global_state["rfe"].transform(features)
                ml_model = global_state["ml_model"]

                prediction_result = int(ml_model.predict(features_opt)[0])
                prob = float(prediction_result)

                if hasattr(ml_model, "predict_proba"):
                    try:
                        prob = float(ml_model.predict_proba(features_opt)[0][1])
                    except Exception as e:
                        if "validate_features" in str(e) and "XGB" in type(ml_model).__name__:
                            dtest = xgboost.DMatrix(features_opt)
                            prob_raw = ml_model.get_booster().predict(dtest)
                            prob = float(prob_raw[0])
                        else:
                            prob = float(prediction_result)
                elif hasattr(ml_model, "decision_function"):
                    df_val = ml_model.decision_function(features_opt)[0]
                    prob = float(1 / (1 + np.exp(-df_val)))

            # เทียบเฉลย Ground Truth
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

            results.append({
                "id": file.filename,
                "filename": file.filename,
                "prediction_class": "Pes Planus (1)" if prediction_result == 1 else "Normal (0)",
                "prediction_code": prediction_result,
                "confidence": prob,
                "ground_truth": "Pes Planus (1)" if gt_label == 1 else ("Normal (0)" if gt_label == 0 else "-"),
                "eval_status": eval_status,
            })

            # เคลียร์ RAM
            if "img_tensor" in locals(): del img_tensor
            if "features" in locals(): del features
            if "features_opt" in locals(): del features_opt
            if "outputs" in locals(): del outputs

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")