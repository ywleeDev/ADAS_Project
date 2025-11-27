import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# ==============================================================================
# 🏆 [최종 필살기] 분야별 전문가 채용 (Class-wise Ensemble)
# ==============================================================================
# 1. 자동차(0), 자전거(2) 담당: RT-DETR (전체 1등)
MODEL_MAIN_PATH = 'ADAS_Project/models/rtdetr_best.pt'

# 2. 보행자(1) 담당: YOLOv11x (고해상도라 작은 사람 잘 잡음)
# (만약 11x가 없으면 yolov11m_best.pt로 바꾸세요)
MODEL_SUB_PATH  = 'ADAS_Project/models/yolov11x_best.pt'

IMG_DIR = "datasets/kitti/images/val"
LBL_DIR = "datasets/kitti/labels/val"
# ==============================================================================

def load_yolo_label(label_path, img_w, img_h):
    boxes, labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                x, y, w, h = parts[1], parts[2], parts[3], parts[4]
                x1 = (x - w/2) * img_w
                y1 = (y - h/2) * img_h
                x2 = (x + w/2) * img_w
                y2 = (y + h/2) * img_h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
    return torch.tensor(boxes), torch.tensor(labels)

def run_class_wise_ensemble():
    print(f"🚀 [Class-wise Ensemble] 섞지 않고 '잘하는 것'만 골라 담기 시작...")
    
    if not os.path.exists(MODEL_MAIN_PATH) or not os.path.exists(MODEL_SUB_PATH):
        print("❌ 모델 파일이 없습니다. 경로를 확인하세요.")
        return

    model_main = YOLO(MODEL_MAIN_PATH)
    model_sub  = YOLO(MODEL_SUB_PATH)
    
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    
    print(f"\n📊 총 {len(img_files)}장 정밀 분석 중...")

    for img_file in tqdm(img_files):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, img_file.replace('.png', '.txt'))
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        # 각자 추론 (TTA 끄고 순정 실력으로)
        res_main = model_main.predict(img, verbose=False, augment=False)[0]
        res_sub  = model_sub.predict(img, verbose=False, augment=False)[0]

        final_boxes = []
        final_scores = []
        final_labels = []

        # [전략 핵심] 헤쳐 모여!
        
        # 1. RT-DETR에서는 '차(0)'와 '자전거(2)'만 가져옴
        if len(res_main.boxes) > 0:
            for box, score, cls in zip(res_main.boxes.xyxy, res_main.boxes.conf, res_main.boxes.cls):
                cls_id = int(cls.item())
                if cls_id in [0, 2]: # Car, Cyclist
                    final_boxes.append(box.cpu().numpy())
                    final_scores.append(score.item())
                    final_labels.append(cls_id)

        # 2. YOLOv11x에서는 '보행자(1)'만 가져옴
        if len(res_sub.boxes) > 0:
            for box, score, cls in zip(res_sub.boxes.xyxy, res_sub.boxes.conf, res_sub.boxes.cls):
                cls_id = int(cls.item())
                if cls_id == 1: # Pedestrian
                    final_boxes.append(box.cpu().numpy())
                    final_scores.append(score.item())
                    final_labels.append(cls_id)

        # 채점 등록
        preds = []
        if len(final_boxes) > 0:
            preds = [dict(
                boxes=torch.tensor(np.array(final_boxes)), 
                scores=torch.tensor(np.array(final_scores)), 
                labels=torch.tensor(np.array(final_labels)).int()
            )]
        else:
            preds = [dict(boxes=torch.tensor([]), scores=torch.tensor([]), labels=torch.tensor([]))]

        t_boxes, t_labels = load_yolo_label(lbl_path, w, h)
        target = [dict(boxes=t_boxes, labels=t_labels.int())]
        metric.update(preds, target)

    print("\n🧮 최종 성적 산출 중...")
    result = metric.compute()
    
    print("\n" + "="*50)
    print("      🏆 분야별 전문가(Class-wise) 최종 성적 🏆")
    print("="*50)
    print(f"▶ 종합 mAP 0.50  : {result['map_50'].item():.4f}")
    print(f"▶ 정밀 mAP 50-95 : {result['map'].item():.4f}")
    print("-" * 50)
    
    if 'map_50_per_class' in result:
        classes = ['Car', 'Pedestrian', 'Cyclist']
        scores = result['map_50_per_class']
        print("[클래스별 mAP 50]")
        for i, cls in enumerate(classes):
            if i < len(scores):
                print(f"  - {cls:<10} : {scores[i].item():.4f}")
    print("="*50)

if __name__ == "__main__":
    run_class_wise_ensemble()
