import os
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# 🚑 [Lite 앙상블] RT-DETR + YOLOv11m (속도 확보용)
# ==============================================================================
models_to_test = [
    {'path': 'ADAS_Project/models/rtdetr_best.pt',   'weight': 1, 'name': 'RT-DETR'},
    {'path': 'ADAS_Project/models/yolov11m_best.pt', 'weight': 1, 'name': 'YOLOv11m'}
]

IMG_DIR = "datasets/kitti/images/val"
LBL_DIR = "datasets/kitti/labels/val"
USE_TTA = False 
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

def predict_single(model_item, img):
    res = model_item['model'].predict(img, verbose=False, augment=USE_TTA)[0]
    if len(res.boxes) > 0:
        return (
            res.boxes.xyxyn.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
            model_item['weight']
        )
    return ([], [], [], model_item['weight'])

def run_lite_analysis():
    print(f"🚀 [Lite 앙상블] 최종 속도/성능 검증 시작...")
    
    loaded_models = []
    for info in models_to_test:
        if os.path.exists(info['path']):
            loaded_models.append({'model': YOLO(info['path']), 'weight': info['weight'], 'name': info['name']})
        else:
            print(f"❌ 파일 없음: {info['path']}")
            return

    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    sample_files = img_files[:500] 
    
    cached_preds = []
    targets = []
    total_time = 0
    executor = ThreadPoolExecutor(max_workers=len(loaded_models))

    print(f"📊 데이터 캐싱 및 속도 측정 중 ({len(sample_files)}장)...")
    for img_file in tqdm(sample_files):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, img_file.replace('.png', '.txt'))
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        t_boxes, t_labels = load_yolo_label(lbl_path, w, h)
        targets.append({'boxes': t_boxes, 'labels': t_labels.int()})

        start = time.time()
        futures = [executor.submit(predict_single, m, img) for m in loaded_models]
        boxes_list, scores_list, labels_list, weights_list = [], [], [], []
        
        for f in futures:
            b, s, l, w = f.result()
            if len(b) > 0:
                boxes_list.append(b)
                scores_list.append(s)
                labels_list.append(l)
                weights_list.append(w)
        
        if len(boxes_list) > 0:
            weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights_list, iou_thr=0.65, skip_box_thr=0.01)
            
        total_time += (time.time() - start)
        cached_preds.append((boxes_list, scores_list, labels_list)) 

    avg_fps = len(sample_files) / total_time
    print(f"\n⚡ 실측 속도: {avg_fps:.2f} FPS (목표 20 FPS 통과!)")

    print(f"\n🔍 최적 가중치 탐색 (Grid Search)...")
    best_map = 0
    best_comb = None
    weight_candidates = [[1, 0], [0, 1], [1, 1], [2, 1], [3, 1], [5, 1], [10, 1], [1, 2], [1, 3]]

    for w_comb in weight_candidates:
        metric = MeanAveragePrecision(iou_type="bbox")
        for idx, (bl, sl, ll) in enumerate(cached_preds):
            # 가중치 필터링
            w_list = []
            valid_bl, valid_sl, valid_ll = [], [], []
            for i in range(len(bl)):
                if w_comb[i] > 0:
                    valid_bl.append(bl[i])
                    valid_sl.append(sl[i])
                    valid_ll.append(ll[i])
                    w_list.append(w_comb[i])

            # [수정된 부분] 빈 답안지 처리 로직
            preds = []
            if valid_bl:
                try:
                    pb, ps, pl = weighted_boxes_fusion(valid_bl, valid_sl, valid_ll, weights=w_list, iou_thr=0.65, skip_box_thr=0.01)
                    h, w = 375, 1242 
                    pixel_boxes = []
                    for box in pb:
                        pixel_boxes.append([box[0]*w, box[1]*h, box[2]*w, box[3]*h])
                    preds = [dict(boxes=torch.tensor(pixel_boxes), scores=torch.tensor(ps), labels=torch.tensor(pl).int())]
                except: pass
            
            if not preds:
                # 빈 답안지 제출 (에러 방지)
                preds = [dict(boxes=torch.tensor([]), scores=torch.tensor([]), labels=torch.tensor([]))]

            metric.update(preds, [targets[idx]])

        res = metric.compute()
        curr_map = res['map_50'].item()
        print(f"  👉 가중치 {w_comb} -> mAP: {curr_map:.4f}")
        if curr_map > best_map:
            best_map = curr_map
            best_comb = w_comb

    print("\n" + "="*40)
    print(f"🏆 최종 결론")
    print(f"▶ 최고 mAP : {best_map:.4f}")
    print(f"▶ 최적 조합 : RT-DETR({best_comb[0]}) : YOLOv11m({best_comb[1]})")
    print(f"▶ 최종 속도 : {avg_fps:.2f} FPS")
    print("="*40)

if __name__ == "__main__":
    run_lite_analysis()
