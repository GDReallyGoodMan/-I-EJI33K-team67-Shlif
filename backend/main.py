from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import traceback  # Для вывода ошибок в терминал
import glob
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import base64
from backend.model_loader import model_manager, DEVICE

app = FastAPI(title="GeoVision API Premium")

# Папки для Active Learning
ANNOTATIONS_DIR = Path("dataset/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.on_event("startup")
async def startup_event():
    model_manager.load_model("weights/best_multitask_unet.pth")

@app.post("/api/analyze")
async def analyze_panorama(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        model = model_manager.get_model()
        result = process_image(contents, file.filename, model)
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"Ошибка сервера: {str(e)}"}, status_code=500)

@app.post("/api/save_mask")
async def save_mask(request: Request):
    """Сохранение нарисованной геологической маски"""
    try:
        data = await request.json()
        image_id = data.get("imageId", "unknown")
        mask_b64 = data.get("maskBase64", "").split(",")[-1]
        
        mask_data = base64.b64decode(mask_b64)
        mask_path = ANNOTATIONS_DIR / f"{image_id}_gt.png"
        with open(mask_path, "wb") as f:
            f.write(mask_data)
        return {"status": "success", "message": "Маска успешно сохранена в датасет!"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/retrain")
async def retrain_model():
    """Реальный запуск быстрого дообучения (Active Learning) на CPU/GPU"""
    try:
        # 1. Находим самую свежую сохраненную маску
        mask_files = glob.glob(str(ANNOTATIONS_DIR / "*_gt.png"))
        if not mask_files:
            return JSONResponse(
                content={"error": "Не найдено сохраненных масок для обучения. Сначала нарисуйте и сохраните маску в редакторе."}, 
                status_code=400
            )
        
        latest_mask_path = Path(max(mask_files, key=os.path.getmtime))
        image_id = latest_mask_path.name.replace("_gt.png", "")
        
        # Находим соответствующий оригинал панорамы в кэше результатов
        original_image_path = Path("frontend/static/results") / f"{image_id}_1_orig.jpg"
        if not original_image_path.exists():
            return JSONResponse(
                content={"error": f"Оригинальное изображение {image_id} не найдено в кэше результатов."}, 
                status_code=400
            )
                    # 2. Загружаем оригинал и маску
        img = cv2.imread(str(original_image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_img = cv2.imread(str(latest_mask_path))
        # Переводим маску в бинарную (ищем любые не черные пиксели)
        gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        _, talc_mask = cv2.threshold(gray_mask, 10, 1, cv2.THRESH_BINARY)
        
        h, w = img.shape[:2]
        
        # Защита от мелких тестовых картинок
        if h < 512 or w < 512:
            img = cv2.resize(img, (512, 512))
            talc_mask = cv2.resize(talc_mask, (512, 512))
            h, w = 512, 512

        # 3. Нарезаем на патчи 512x512
        patches_img = []
        patches_mask = []
        patch_size = 512
        
        for y in range(0, h - patch_size + 1, 256):
            for x in range(0, w - patch_size + 1, 256):
                p_mask = talc_mask[y:y+patch_size, x:x+patch_size]
                # Берем только те патчи, где есть маска, нарисованная пользователем
                if np.sum(p_mask) > 10: 
                    p_img = img[y:y+patch_size, x:x+patch_size]
                    patches_img.append(p_img)
                    patches_mask.append(p_mask)
                    
        # Если патчи не нарезались, берем центр
        if not patches_img:
            cy, cx = h // 2, w // 2
            patches_img.append(img[cy-256:cy+256, cx-256:cx+256])
            patches_mask.append(talc_mask[cy-256:cy+256, cx-256:cx+256])

        # 4. Нормализация и перевод в тензоры PyTorch
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        x_tensors = []
        y_tensors = []
        for pi, pm in zip(patches_img, patches_mask):
            pi_norm = (pi / 255.0 - mean) / std
            pi_tensor = torch.tensor(pi_norm, dtype=torch.float32).permute(2, 0, 1).to(DEVICE)
            pm_tensor = torch.tensor(pm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_tensors.append(pi_tensor)
            y_tensors.append(pm_tensor)
            
        X = torch.stack(x_tensors)
        Y = torch.stack(y_tensors)

        # 5. Быстрое дообучение (Fine-Tuning)
        model = model_manager.get_model()
        model.train()  # Режим обучения
        
        # Замораживаем тяжелый энкодер для скорости
        for param in model.encoder.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # 5 быстрых эпох декодера (займет пару секунд)
        for epoch in range(5):
            optimizer.zero_grad()
            pred_masks, _ = model(X)
            loss = criterion(pred_masks, Y)
            loss.backward()
            optimizer.step()
            print(f"[Active Learning] Эпоха {epoch+1}/5 | Loss: {loss.item():.4f}")
            
        # 6. Сохраняем новые веса
        torch.save(model.state_dict(), "weights/best_multitask_unet.pth")
        
        # Возвращаем модель в режим инференса
        model.eval()
        
        return {
            "status": "success", 
            "message": f"Обучение завершено! Модель успешно адаптирована под новые данные {image_id}. Веса обновлены."
        }
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"Ошибка при дообучении: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)