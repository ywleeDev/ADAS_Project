import os
import time
import torch
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# 🏆 [최종 병기] RT-DETR 단독 + TTA
MODEL_PATH = 'ADAS_Project/models/rtdetr_best.pt'
IMG_DIR = "datasets/kitti/images/val"
LBL_DIR = "datasets/kitti/labels/val"

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

def run_tta_test():
    print(f"🚀 [Final Strategy] RT-DETR + TTA(Augmentation) 평가 시작...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    metric = MeanAveragePrecision(iou_type="bbox")
    
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    # 전체 데이터로 확실하게 검증
    
    print(f"\n📊 전체 {len(img_files)}장 TTA 추론 및 속도 측정 중...")
    
    total_time = 0
    
    for img_file in tqdm(img_files):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, img_file.replace('.png', '.txt'))
        
        # 이미지 로드 및 정답 로드
        # (cv2 imread는 I/O 시간이므로 속도 측정에서 제외하는게 맞으나, 
        #  전체 파이프라인 속도를 보수적으로 잡기 위해 포함해도 무방)
        img = cv2.imread(img_path) 
        h, w, _ = img.shape
        
        t_boxes, t_labels = load_yolo_label(lbl_path, w, h)
        target = [dict(boxes=t_boxes, labels=t_labels.int())]

        # [핵심] 시간 측정 시작
        start = time.time()
        
        # augment=True: 이게 바로 TTA입니다. (속도는 느려지지만 정확도 상승)
        results = model.predict(img, verbose=False, augment=True)[0]
        
        total_time += (time.time() - start)

        # 결과 변환
        if len(results.boxes) > 0:
            pred_boxes = results.boxes.xyxy.cpu() # 픽셀 좌표
            pred_scores = results.boxes.conf.cpu()
            pred_labels = results.boxes.cls.cpu().int()
            
            preds = [dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)]
        else:
            preds = [dict(boxes=torch.tensor([]), scores=torch.tensor([]), labels=torch.tensor([]))]
            
        metric.update(preds, target)

    # 결과 출력
    print("\n🧮 최종 점수 계산 중...")
    res = metric.compute()
    
    avg_fps = len(img_files) / total_time
    
    print("\n" + "="*50)
    print("      🏆 RT-DETR + TTA 최종 성적표 🏆")
    print("="*50)
    print(f"▶ mAP 0.50 (정확도) : {res['map_50'].item():.4f}")
    print(f"▶ mAP 50-95 (정밀도): {res['map'].item():.4f}")
    print(f"▶ 추론 속도 (FPS)   : {avg_fps:.2f} FPS")
    print("-" * 50)
    
    if avg_fps >= 20:
        print("🎉 [합격] 속도 20 FPS 이상 유지 성공!")
        print("   이 점수와 속도를 최종 결과로 제출하세요.")
    else:
        print("⚠️ [경고] 속도가 20 FPS 미만입니다.")
        print("   augment=False로 끄고(기본 모드) 제출해야 합니다.")
    print("="*50)

if __name__ == "__main__":
    import cv2 # 런타임 임포트 에러 방지
    run_tta_test()
