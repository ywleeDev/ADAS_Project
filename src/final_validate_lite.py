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
# 🚑 [긴급 수정] 가벼운 앙상블 (RT-DETR + YOLOv11m)
# ==============================================================================
models_to_test = [
    {'path': 'ADAS_Project/models/rtdetr_best.pt',   'weight': 1, 'name': 'RT-DETR'},
    {'path': 'ADAS_Project/models/yolov11m_best.pt', 'weight': 1, 'name': 'YOLOv11m'}
]

IMG_DIR = "datasets/kitti/images/val"
LBL_DIR = "datasets/kitti/labels/val"
USE_TTA = False # 속도 확보를 위해 끔 (이게 켜지면 느려짐)
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
    # 각 모델 추론 (병렬 처리용 함수)
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
    print(f"🚀 [Lite 앙상블] RT-DETR + YOLOv11m 속도/성능 측정 시작...")
    
    loaded_models = []
    for info in models_to_test:
        if os.path.exists(info['path']):
            loaded_models.append({'model': YOLO(info['path']), 'weight': info['weight'], 'name': info['name']})
            print(f"  ✅ 로드 완료: {info['name']}")
        else:
            print(f"  ❌ 파일 없음: {info['path']}")
            return

    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    sample_files = img_files[:500] # 500장 샘플링
    print(f"\n📊 500장 샘플링 테스트 중...")
    
    cached_preds = []
    targets = []
    total_time = 0
    
    # 병렬 처리를 위한 실행기 생성
    executor = ThreadPoolExecutor(max_workers=len(loaded_models))

    for img_file in tqdm(sample_files):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, img_file.replace('.png', '.txt'))
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        t_boxes, t_labels = load_yolo_label(lbl_path, w, h)
        targets.append({'boxes': t_boxes, 'labels': t_labels.int()})

        start = time.time()
        
        # 병렬 추론 실행
        futures = [executor.submit(predict_single, m, img) for m in loaded_models]
        boxes_list, scores_list, labels_list, weights_list = [], [], [], []
        
        for f in futures:
            b, s, l, w = f.result()
            if len(b) > 0:
                boxes_list.append(b)
                scores_list.append(s)
                labels_list.append(l)
                weights_list.append(w)
        
        # WBF 실행 (시간 측정 포함)
        if len(boxes_list) > 0:
            weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights_list, iou_thr=0.65, skip_box_thr=0.01)
            
        total_time += (time.time() - start)
        cached_preds.append((boxes_list, scores_list, labels_list)) # 결과 캐싱

    avg_fps = len(sample_files) / total_time
    print(f"\n⚡ 교체 후 실측 속도: {avg_fps:.2f} FPS")
    
    if avg_fps >= 20:
        print("🎉 [통과] 목표 속도(20 FPS)를 달성했습니다!")
    else:
        print("⚠️ [경고] 아직도 20 FPS 미만입니다. 단일 모델 제출을 고려하세요.")

    # 최적 가중치 탐색
    print(f"\n🔍 최적 가중치 탐색 (Grid Search)...")
    best_map = 0
    best_comb = None
    # RT-DETR : YOLOv11m 비율 조합
    weight_candidates = [[1, 0], [0, 1], [1, 1], [2, 1], [3, 1], [5, 1], [10, 1], [1, 2]]

    for w_comb in weight_candidates:
        metric = MeanAveragePrecision(iou_type="bbox")
        for idx, (bl, sl, ll) in enumerate(cached_preds):
            # 가중치 적용하여 재계산
            w_list = []
            valid_bl, valid_sl, valid_ll = [], [], []
            
            # 모델별 결과가 있는지 확인하고 가중치 매핑
            # bl 리스트 순서: 0번(RT-DETR), 1번(YOLOv11m)
            for i in range(len(bl)):
                if w_comb[i] > 0: # 가중치가 0인 모델은 제외
                    valid_bl.append(bl[i])
                    valid_sl.append(sl[i])
                    valid_ll.append(ll[i])
                    w_list.append(w_comb[i])

            if not valid_bl:
                metric.update([], [targets[idx]])
                continue

            pb, ps, pl = weighted_boxes_fusion(valid_bl, valid_sl, valid_ll, weights=w_list, iou_thr=0.65, skip_box_thr=0.01)
            
            # 픽셀 변환 (약식)
            h, w = 375, 1242 
            pixel_boxes = []
            for box in pb:
                pixel_boxes.append([box[0]*w, box[1]*h, box[2]*w, box[3]*h])
            
            preds = [dict(boxes=torch.tensor(pixel_boxes), scores=torch.tensor(ps), labels=torch.tensor(pl).int())]
            metric.update(preds, [targets[idx]])

        res = metric.compute()
        curr_map = res['map_50'].item()
        print(f"  👉 가중치 {w_comb} -> mAP: {curr_map:.4f}")
        if curr_map > best_map:
            best_map = curr_map
            best_comb = w_comb

    print("\n" + "="*40)
    print(f"🏆 최종 결론 (Lite Ensemble)")
    print(f"▶ 최고 mAP : {best_map:.4f}")
    print(f"▶ 최적 조합 : RT-DETR({best_comb[0]}) : YOLOv11m({best_comb[1]})")
    print("="*40)

if __name__ == "__main__":
    run_lite_analysis()
