import asyncio
import gc
import io
import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms
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

# ==============================================================
# ⚙️ การตั้งค่า (ปรับผ่าน Environment Variable ได้ ไม่ต้องแก้โค้ด)
# ==============================================================
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/littlepakan/PesPlanusAI2/releases/download/v1.0.0/ens_arch_densenet201.pt",
)
MODEL_PATH = os.getenv("MODEL_PATH", "ens_arch_densenet201.pt")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))                  # ต้องตรงกับตอนเทรน
USE_MEDIAN_FILTER = os.getenv("USE_MEDIAN_FILTER", "1") == "1"  # ต้องตรงกับตอนเทรน
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))              # threshold ของคลาส Pes Planus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_lock = asyncio.Lock()
global_state = {
    "models": None,   # list ของ DenseNet201 (1 ตัว หรือหลายตัวถ้าเป็น ensemble)
    "csv_key": None,
    "gt_map": {},
}


@app.get("/")
def read_root():
    return {"message": "Pes Planus API is running perfectly!"}


# ==============================================================
# 📦 โหลดโมเดล DenseNet201
# ==============================================================
def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    tmp_path = MODEL_PATH + ".part"
    try:
        print(f"⬇️  กำลังดาวน์โหลดโมเดลจาก {MODEL_URL} ...")
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, MODEL_PATH)
        print("✅ ดาวน์โหลดโมเดลเสร็จแล้ว")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _clean_state_dict(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        for prefix in ("module.", "model.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def _is_state_dict(obj) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())


def _collect(obj) -> list:
    """ดึง nn.Module หรือ state_dict ทุกอันออกจากไฟล์ .pt (รองรับ ensemble หลายโมเดล)"""
    if isinstance(obj, nn.Module):
        return [obj]
    if _is_state_dict(obj):
        return [obj]
    items = []
    if isinstance(obj, (list, tuple)):
        for o in obj:
            items += _collect(o)
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, (dict, list, tuple, nn.Module)):
                items += _collect(v)
    return items


def _build_densenet201_from_state(sd: dict) -> nn.Module:
    sd = _clean_state_dict(sd)
    if "features.conv0.weight" not in sd:
        raise ValueError("state_dict นี้ไม่ใช่ของ DenseNet")

    w_key = next((k for k in sd if k.startswith("classifier") and k.endswith("weight")), None)
    if w_key is None:
        raise ValueError("ไม่พบ classifier weight ใน state_dict")

    num_out = sd[w_key].shape[0]
    net = models.densenet201(weights=None)
    if w_key == "classifier.weight":
        net.classifier = nn.Linear(1920, num_out)
    elif w_key == "classifier.1.weight":
        net.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1920, num_out))
    else:
        raise ValueError(f"โครงสร้าง classifier ไม่รองรับ: {w_key}")

    net.load_state_dict(sd, strict=True)
    return net


def load_models() -> list:
    download_model_if_needed()

    try:
        obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except Exception:
        # ไฟล์ที่เซฟทั้งโมเดล (torch.save(model)) ต้องใช้ weights_only=False
        # ⚠️ ใช้กับไฟล์ที่เชื่อถือได้เท่านั้น
        obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    loaded = []
    for item in _collect(obj):
        if isinstance(item, nn.Module):
            net = item
        else:
            try:
                net = _build_densenet201_from_state(item)
            except ValueError:
                continue  # ข้าม state_dict ที่ไม่ใช่โมเดล (เช่น optimizer)
        loaded.append(net.to(device).eval())

    if not loaded:
        raise RuntimeError("ไม่พบโมเดล DenseNet201 ในไฟล์ .pt")
    print(f"✅ โหลดโมเดลสำเร็จ {len(loaded)} ตัว บน {device}")
    return loaded


# ==============================================================
# 🖼️ Pre-processing
# ==============================================================
def apply_median_filter(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    median_img = cv2.medianBlur(img_cv, 3)
    return Image.fromarray(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))


def get_transforms():
    steps = []
    if USE_MEDIAN_FILTER:
        steps.append(transforms.Lambda(apply_median_filter))
    steps += [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


_transform = get_transforms()


# ==============================================================
# 📄 CSV เฉลย
# ==============================================================
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


# ==============================================================
# 🟢 Endpoint สำหรับรับภาพมาวิเคราะห์
# ==============================================================
@app.post("/api/predict")
async def predict_single_image(
    file: UploadFile = File(...),
    gt_option: str = Form("none"),
    csv_file: Optional[UploadFile] = File(None),
):
    try:
        async with model_lock:
            # 1. โหลดโมเดล DenseNet201 (ครั้งแรกครั้งเดียว)
            if global_state["models"] is None:
                try:
                    global_state["models"] = await asyncio.to_thread(load_models)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"โหลดโมเดลไม่สำเร็จ: {str(e)}")

            # 2. จัดการไฟล์ CSV (ถ้าอัปโหลดมา)
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
        img_tensor = _transform(image).unsqueeze(0).to(device)

        # 3. ทำนาย (เฉลี่ยความน่าจะเป็นถ้ามีหลายโมเดล)
        probs = []
        with torch.no_grad():
            for net in global_state["models"]:
                logits = net(img_tensor)
                if logits.shape[1] == 1:
                    p = torch.sigmoid(logits)[:, 0]
                else:
                    p = torch.softmax(logits, dim=1)[:, 1]
                probs.append(float(p.item()))
        prob_class_1 = float(np.mean(probs))

        prediction_result = int(prob_class_1 >= THRESHOLD)
        final_confidence = prob_class_1 if prediction_result == 1 else 1.0 - prob_class_1

        # 4. เทียบกับเฉลย
        fname = str(file.filename).strip().lower()
        bname = os.path.splitext(fname)[0]
        gt_label = global_state["gt_map"].get(fname)
        if gt_label is None:
            gt_label = global_state["gt_map"].get(bname)

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
            "confidence": final_confidence,
            "ground_truth": "Pes Planus (1)" if gt_label == 1 else ("Normal (0)" if gt_label == 0 else "-"),
            "eval_status": eval_status,
        }

        del img_tensor
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
