"""
工业质检系统 - FastAPI 后端
提供图片上传接口，调用 PatchCore 推理，返回检测结果 + 热力图
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import base64
import tempfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.patchcore import PatchCore

# ── 配置 ──────────────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\industrial_ML\models\patchcore_bottle.pkl"
UPLOAD_DIR  = r"D:\industrial_ML\reports\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="工业视觉质检系统", version="1.0.0")

BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── 模型加载 (只在启动时加载一次) ──────────────────────────────────────────
print("[App] Loading PatchCore model...")
model = PatchCore()
if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)
    print("[App] Model loaded successfully.")
else:
    print(f"[App] WARNING: Model file not found at {MODEL_PATH}")
    print("[App] Please run 'python src/patchcore.py' first to train the model.")


def generate_heatmap_overlay(original_img_path: str, heatmap: np.ndarray) -> str:
    """
    将异常热力图叠加在原始图片上，返回 base64 编码的 PNG 图片字符串
    """
    # 读取原始图片
    orig = cv2.imread(original_img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w = orig.shape[:2]

    # 将热力图 resize 到原图大小
    hmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))

    # 归一化到 0-255
    hmap_norm = (hmap_resized - hmap_resized.min()) / (hmap_resized.max() - hmap_resized.min() + 1e-8)
    hmap_color = (cm.jet(hmap_norm)[:, :, :3] * 255).astype(np.uint8)

    # 叠加: 原图 60% + 热力图 40%
    overlay = (orig * 0.6 + hmap_color * 0.4).astype(np.uint8)

    # 转换为 base64
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64


def encode_original_image(img_path: str) -> str:
    """将原始图片编码为 base64"""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── API 路由 ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """接收上传图片，返回检测结果"""

    # 验证文件类型
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        raise HTTPException(status_code=400, detail="Only image files (png, jpg, jpeg, bmp, tiff) are supported.")

    # 保存到临时文件
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if model.memory_bank is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train PatchCore first.")

        # 推理
        is_pass, score, heatmap, confidence_pct = model.predict(tmp_path)

        # 生成热力图叠加图
        overlay_b64 = generate_heatmap_overlay(tmp_path, heatmap)
        original_b64 = encode_original_image(tmp_path)

        return JSONResponse({
            "result":       "PASS" if is_pass else "FAIL",
            "is_pass":      bool(is_pass),
            "score":        round(float(score), 4),
            "threshold":    round(float(model.threshold), 4),
            "confidence":   round(float(confidence_pct), 1),
            "original_img": original_b64,
            "heatmap_img":  overlay_b64,
            "filename":     file.filename,
        })

    finally:
        # 推理完成后删除临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model.memory_bank is not None,
        "threshold": round(float(model.threshold), 4) if model.threshold else None,
        "memory_bank_size": len(model.memory_bank) if model.memory_bank is not None else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
