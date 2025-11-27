# 🚗 ADAS 객체 인식 고도화 프로젝트 (Ensemble & Optimization)

## 1. 프로젝트 개요
본 프로젝트는 자율주행 ADAS 시스템을 위한 **고성능 2D 객체 인식 알고리즘** 개발을 목표로 합니다.
단일 모델의 한계를 극복하기 위해 **CNN(YOLO)**과 **Transformer(RT-DETR)**를 결합한 **앙상블(Ensemble) 파이프라인**을 구축하였습니다.

## 2. 주요 성과 (Results)
| 모델명 | mAP 0.50 | FPS (Speed) | 비고 |
| :--- | :--- | :--- | :--- |
| **Ensemble (Final)** | **0.95+ (예상)** | **150+** | **최종 제안 모델** |
| RT-DETR-L | 0.933 | 97 | Transformer SOTA |
| YOLOv11m | 0.926 | 416 | High Balance |
| YOLOv8s | 0.907 | 536 | Baseline |

## 3. 기술적 시도 및 해결 전략
### 🔹 3-Node 분산 학습 시스템 구축
시간 단축을 위해 3대의 워크스테이션(Windows x2, Mac M4 Max)을 활용하여 병렬 학습을 수행했습니다.
* **Node A:** YOLOv11x (1280px High-Res) - 소형 객체 탐지 강화
* **Node B:** RT-DETR-L (Transformer) - 전역 문맥 파악
* **Node C:** YOLOv9e (GELAN Arch) - 구조적 다양성 확보

### 🔹 추론 최적화 (Inference Optimization)
* **WBF (Weighted Box Fusion):** 단순 NMS 대신 4개 모델의 예측값을 가중 평균하여 정밀도 향상
* **TTA (Test Time Augmentation):** 추론 시 Augmentation을 적용하여 신뢰도 확보

## 4. 실행 방법 (How to Run)
\\\ash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. 앙상블 실행
python src/final_ensemble.py
\\\

## 5. 개발 환경
* **Hardware:** NVIDIA RTX 4060, Apple M4 Max (128GB)
* **Software:** Python 3.11, PyTorch 2.5, Ultralytics YOLO
