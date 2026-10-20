import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import os
import cv2
import time
import datetime
import uuid
import math
import shutil
import logging
import base64

nest_asyncio.apply()

app = FastAPI(title="SmartBin")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/heatmaps", exist_ok=True)
os.makedirs("collected_data/auto_accept", exist_ok=True)
os.makedirs("collected_data/manual_review", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = 'app/weights/best.pt'
if not os.path.exists(model_path):
    model_path = 'app/weights/best.pt'
model = YOLO(model_path)

DECISION_THRESHOLD = 0.53
TEMPERATURE_T = 0.9359

RHO_TABLE = {
    "SmartBin-carbs": 0.65, 
    "SmartBin-proteins": 0.85, 
    "SmartBin-vege_fruits": 0.55, 
    "SmartBin-Sauce_soup": 1.05, 
    "SmartBin-mixed_and_waste": 1.00
}

YOLO_TO_API_MAP = {
    "SmartBin-carbs": "rice_rice_dishes",
    "SmartBin-proteins": "chicken",
    "SmartBin-vege_fruits": "vegetables_cooked",
    "SmartBin-Sauce_soup": "soup_curry",
    "SmartBin-mixed_and_waste": "mixed_leftovers"
}

def apply_ts_aurc_calibration(raw_conf, temperature):
    if raw_conf >= 1.0: return 0.9999
    if raw_conf <= 0.0: return 0.0001
    logit = math.log(raw_conf / (1.0 - raw_conf))
    scaled_logit = logit / temperature
    return 1.0 / (1.0 + math.exp(-scaled_logit))

@app.get("/")
async def root_health_check():
    return {
        "status": "online", 
        "system": "SmartBin v1.0",
        "dataset_base": "1774_images"
    }

@app.post("/ai/infer")
async def execute_waste_inference(
    image_file: UploadFile = File(...),
    total_scale_weight_grams: float = Form(...)
):
    start_time = time.time()
    event_id = f"E{datetime.datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    
    # weight validation
    if total_scale_weight_grams < 0:
        raise HTTPException(status_code=400, detail="Invalid weight: Must be greater than or equal to 0.")
        
    try:
        temp_img_path = f"temp_uploads/{event_id}.jpg"
        with open(temp_img_path, "wb") as buffer:
            buffer.write(await image_file.read())
            
        results = model(temp_img_path)
        result = results[0]
        
        # mask overlay
        heatmap_img = result.plot()
        heatmap_filename = f"{event_id}_gradcam.jpg"
        cv2.imwrite(f"./static/heatmaps/{heatmap_filename}", heatmap_img)
        
        _, encoded_img = cv2.imencode('.jpg', heatmap_img)
        heatmap_base64 = base64.b64encode(encoded_img).decode('utf-8')
        
        detected_items = []
        total_mass_factor = 0.0
        
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                class_name = model.names[int(result.boxes.cls[i].item())]
                raw_conf = float(result.boxes.conf[i].item())
                pixel_count = int(mask.sum().item())
                calibrated_item_conf = apply_ts_aurc_calibration(raw_conf, TEMPERATURE_T)
                rho = RHO_TABLE.get(class_name, 1.0)
                mass_factor = pixel_count * rho
                total_mass_factor += mass_factor
                
                detected_items.append({
                    "class_name": class_name, "calib_conf": calibrated_item_conf, 
                    "pixels": pixel_count, "mass_factor": mass_factor, "rho": rho
                })
        
        # calculate confidence score
        if detected_items:
            image_calibrated_conf = sum(item["calib_conf"] for item in detected_items) / len(detected_items)
        else:
            image_calibrated_conf = 0.0
        
        action_decision = "AUTO_ACCEPT" if image_calibrated_conf >= DECISION_THRESHOLD else "MANUAL_REVIEW"
        
        archive_dir = "collected_data/auto_accept" if action_decision == "AUTO_ACCEPT" else "collected_data/manual_review"
        shutil.move(temp_img_path, f"{archive_dir}/{event_id}.jpg")
        
        breakdown = {}
        total_pixels = sum(item["pixels"] for item in detected_items)
        
        for item in detected_items:
            api_class = YOLO_TO_API_MAP.get(item["class_name"])
            if api_class and total_mass_factor > 0:
                est_weight = total_scale_weight_grams * (item["mass_factor"] / total_mass_factor)
                area_pct = (item["pixels"] / total_pixels) * 100 if total_pixels > 0 else 0
                
                if api_class not in breakdown:
                    breakdown[api_class] = {
                        "allocated_pixels": 0,
                        "area_percentage": 0.0,
                        "density_rho": item["rho"],
                        "calculated_weight_grams": 0.0,
                        "confidence_score": round(item["calib_conf"], 4)
                    }
                
                breakdown[api_class]["allocated_pixels"] += item["pixels"]
                breakdown[api_class]["area_percentage"] += round(area_pct, 1)
                breakdown[api_class]["calculated_weight_grams"] += round(est_weight, 2)
        
        end_time = time.time()
        
        return {
            "event_id": event_id,
            "system_metadata": {
                "model_version": "smartbin-yolov11s-seg-v1.0",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latency_ms": round((end_time - start_time) * 1000, 1)
            },
            "hardware_sensor_inputs": {
                "total_scale_weight_grams": total_scale_weight_grams
            },
            "calibration_security_gate": {
                "optimization_method": "TS-AURC",
                "optimized_temperature_parameter": TEMPERATURE_T,
                "calibrated_confidence_score": round(image_calibrated_conf, 4),
                "operational_thresholds": {
                    "decision_threshold": DECISION_THRESHOLD
                },
                "action_routing_decision": action_decision
            },
            "density_scaled_composition": {
                "total_segmented_food_pixels": total_pixels,
                "composition_metric_type": "density-weighted area-based mass apportionment",
                "breakdown": breakdown
            },
            "explainable_ai_logs": {
                "algorithm_applied": "YOLOv11s-seg Prediction & Mask Overlay",
                "target_backbone_layer": "segmentation_head_output",
                "heatmap_image_url": f"http://127.0.0.1:8001/static/heatmaps/{heatmap_filename}", #backup in local
                "heatmap_base64": heatmap_base64, #software used
                "audit_trail_status": "verified_trustworthy"
            }
        }
    except Exception as e:
        logging.error(f"Inference failed for {event_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal AI Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)