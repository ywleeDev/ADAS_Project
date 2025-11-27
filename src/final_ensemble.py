import os
import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ==============================================================================
# 🏆 [최종 병기] 4개 모델 앙상블 설정
# ==============================================================================
# models 폴더 안에 파일 4개가 다 있어야 합니다!
models_info = [
    # 1. RT-DETR (트랜스포머): 0.933 (가중치 4 - 가장 높음)
    {'path': 'models/rtdetr_best.pt',   'weight': 4},
    
    # 2. YOLOv11x (고해상도): 0.925 (가중치 3 - 정밀도 담당)
    {'path': 'models/yolov11x_best.pt', 'weight': 3},
    
    # 3. YOLOv9e (GELAN): 구조적 다양성 (가중치 2)
    {'path': 'models/yolov9e_best.pt',  'weight': 2},
    
    # 4. YOLOv8s (SGD): 기본기 (가중치 1 - 보조)
    {'path': 'models/yolov8s_best.pt',  'weight': 1}
]

# 테스트할 이미지 경로
TEST_IMG_DIR = "datasets/kitti/images/val"
OUTPUT_DIR = "final_result_images"
# ==============================================================================

def run_wbf_ensemble(img_path):
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    img = cv2.imread(img_path)
    if img is None: return None
    h, w, _ = img.shape

    # 각 모델 추론
    for info in models_info:
        # 파일 있는지 확인
        if not os.path.exists(info['path']):
            print(f"⚠️ 파일 없음 (건너뜀): {info['path']}")
            continue
            
        try:
            # 모델 로드 & 추론 (TTA 켜기)
            model = YOLO(info['path'])
            results = model.predict(img, verbose=False, augment=True)
            
            for r in results:
                if len(r.boxes) == 0: continue
                
                # 정규화된 좌표 (0~1)로 변환
                boxes_list.append(r.boxes.xyxyn.cpu().numpy())
                scores_list.append(r.boxes.conf.cpu().numpy())
                labels_list.append(r.boxes.cls.cpu().numpy())
                weights.append(info['weight'])
                
        except Exception as e:
            print(f"🚨 모델 에러 ({info['path']}): {e}")
            continue

    if len(boxes_list) == 0: return img

    # 🌟 WBF 실행 (핵심 기술)
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, 
        weights=weights, iou_thr=0.6, skip_box_thr=0.1
    )

    # 결과 그리기
    for i in range(len(boxes)):
        x1 = int(boxes[i][0] * w)
        y1 = int(boxes[i][1] * h)
        x2 = int(boxes[i][2] * w)
        y2 = int(boxes[i][3] * h)
        
        # 클래스별 색상 (0:차-초록, 1:사람-빨강, 2:자전거-파랑)
        cls_id = int(labels[i])
        score = scores[i]
        
        color = (0, 255, 0) # 기본 초록
        if cls_id == 1: color = (0, 0, 255) # 사람 빨강
        elif cls_id == 2: color = (255, 0, 0) # 자전거 파랑
        
        # 박스랑 글씨 쓰기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{cls_id} {score:.2f}"
        cv2.putText(img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img

if __name__ == "__main__":
    print("🚀 [최종 병기] 4개 모델 앙상블 시작...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 랜덤 20장만 뽑아서 테스트
    if not os.path.exists(TEST_IMG_DIR):
        print(f"🚨 데이터셋 경로 확인 필요: {TEST_IMG_DIR}")
    else:
        img_files = [os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')][:20]
        
        for img_path in img_files:
            fname = os.path.basename(img_path)
            result_img = run_wbf_ensemble(img_path)
            
            if result_img is not None:
                save_path = os.path.join(OUTPUT_DIR, fname)
                cv2.imwrite(save_path, result_img)
                print(f"  -> 앙상블 결과 저장: {save_path}")

        print(f"\n🏆 작업 완료! '{OUTPUT_DIR}' 폴더에 저장된 사진들을 확인하세요.")