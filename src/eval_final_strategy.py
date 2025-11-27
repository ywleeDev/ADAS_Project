import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# ==============================================================================
# 🏆 [전략 A] RT-DETR 독재 모드 (10:1)
# ==============================================================================
models_to_test = [
    # 1. 1등공신 RT-DETR (가중치 10: 내 말이 곧 법이다)
    {'path': 'ADAS_Project/models/rtdetr_best.pt',   'weight': 10},
    
    # 2. 보좌관 YOLOv11x (가중치 1: 혹시 제가 놓친 게 있나요?)
    {'path': 'ADAS_Project/models/yolov11x_best.pt', 'weight': 1}
]

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

def run_evaluation():
    print(f"🚀 [전략 A] RT-DETR 몰아주기(10:1) 평가 시작...")
    
    loaded_models = []
    weights = []
    
    for info in models_to_test:
        if os.path.exists(info['path']):
            print(f"  ✅ 로드: {info['path']} (가중치 {info['weight']})")
            loaded_models.append(YOLO(info['path']))
            weights.append(info['weight'])
        else:
            print(f"  ❌ 파일 없음: {info['path']}")

    if not loaded_models: return

    metric = MeanAveragePrecision(iou_type="bbox")
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    print(f"\n📊 총 {len(img_files)}장 정밀 채점 중...")

    for img_file in tqdm(img_files):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, img_file.replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

        boxes_list, scores_list, labels_list = [], [], []
        
        for model in loaded_models:
            # TTA 켜서 최대한 성능 끌어올림
            res = model.predict(img, verbose=False, augment=True)[0]
            if len(res.boxes) > 0:
                boxes_list.append(res.boxes.xyxyn.cpu().numpy())
                scores_list.append(res.boxes.conf.cpu().numpy())
                labels_list.append(res.boxes.cls.cpu().numpy())
        
        if len(boxes_list) == 0: continue

        # WBF (IoU Threshold를 0.7로 높여서 엄격하게 병합)
        pred_boxes, pred_scores, pred_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights, iou_thr=0.7, skip_box_thr=0.01
        )

        pixel_boxes = []
        for box in pred_boxes:
            pixel_boxes.append([box[0]*w, box[1]*h, box[2]*w, box[3]*h])
        
        preds = [dict(boxes=torch.tensor(pixel_boxes), scores=torch.tensor(pred_scores), labels=torch.tensor(pred_labels).int())]
        target_boxes, target_labels = load_yolo_label(lbl_path, w, h)
        target = [dict(boxes=target_boxes, labels=target_labels.int())]
        
        metric.update(preds, target)

    print("\n🧮 계산 완료! 성적표 출력 중...")
    result = metric.compute()
    print("="*40)
    print(f"🏆 [전략 A] 최종 mAP 0.50 : {result['map_50'].item():.4f}")
    print(f"   (RT-DETR 단독 점수: 0.9330)")
    print("="*40)
    
    if result['map_50'].item() > 0.933:
        print("🎉 성공! 앙상블이 단독 모델을 이겼습니다! 이 결과를 제출하세요.")
    else:
        print("📉 실패... RT-DETR 단독 모델이 더 낫습니다. 앙상블 하지 말고 단독 제출하세요.")

if __name__ == "__main__":
    run_evaluation()
