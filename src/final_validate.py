import os
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from concurrent.futures import ThreadPoolExecutor # 병렬 처리를 위한 도구

# ==============================================================================
# 🚀 [최종 병기] 병렬 처리(Multi-threading) 적용
# ==============================================================================
models_to_test = [
    {'path': 'ADAS_Project/models/rtdetr_best.pt',   'weight': 1}, # 최적 비율 1
    {'path': 'ADAS_Project/models/yolov11x_best.pt', 'weight': 1}  # 최적 비율 1
]

IMG_DIR = "datasets/kitti/images/val"
# ==============================================================================

# 모델 미리 로드 (전역 변수)
loaded_models = []
for info in models_to_test:
    if os.path.exists(info['path']):
        loaded_models.append({'model': YOLO(info['path']), 'weight': info['weight']})
        print(f"✅ 모델 준비 완료: {info['path']}")

def predict_single(model_item, img):
    # 각 모델이 개별 스레드에서 실행될 함수
    res = model_item['model'].predict(img, verbose=False, augment=False)[0]
    if len(res.boxes) > 0:
        return (
            res.boxes.xyxyn.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
            model_item['weight']
        )
    return ([], [], [], model_item['weight'])

def run_speed_test():
    print(f"\n🚀 병렬 처리(Multi-threading) 속도 측정 시작...")
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')][:500]
    
    total_time = 0
    
    # 스레드 풀 생성 (모델 개수만큼 워커 생성)
    executor = ThreadPoolExecutor(max_workers=len(loaded_models))

    for img_file in img_files:
        img_path = os.path.join(IMG_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        
        start = time.time()
        
        # [핵심] 병렬로 추론 던지기
        futures = [executor.submit(predict_single, m, img) for m in loaded_models]
        
        boxes_list, scores_list, labels_list, weights_list = [], [], [], []
        
        # 결과 수집
        for f in futures:
            b, s, l, w = f.result()
            if len(b) > 0:
                boxes_list.append(b)
                scores_list.append(s)
                labels_list.append(l)
                weights_list.append(w)
        
        # WBF (여기는 아주 빠름)
        if len(boxes_list) > 0:
            weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights_list, iou_thr=0.65, skip_box_thr=0.01)
            
        total_time += (time.time() - start)

    avg_fps = len(img_files) / total_time
    print(f"\n⚡ 병렬 처리 적용 후 속도: {avg_fps:.2f} FPS")
    
    if avg_fps >= 20:
        print("🎉 축하합니다! 목표(20 FPS)를 달성했습니다!")
    else:
        print("⚠️ 여전히 20 FPS 미만입니다. 하드웨어 한계일 수 있습니다.")

if __name__ == "__main__":
    if loaded_models:
        run_speed_test()
    else:
        print("❌ 로드된 모델이 없습니다.")