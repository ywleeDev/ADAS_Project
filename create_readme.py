import os

# 깃허브 대문에 들어갈 내용 (완벽한 마크다운 포맷)
readme_content = """# 🚗 ADAS 객체 인식 고도화 및 최적 모델 선정 (ADAS Object Detection Optimization)

## 1. 프로젝트 개요 (Overview)
본 프로젝트는 자율주행 ADAS 시스템을 위한 **고성능 2D 객체 인식 알고리즘 개발**을 목표로 합니다.
KITTI 데이터셋을 기반으로 **① 극한의 정확도(mAP 0.95 목표)**와 **② 실시간성(20 FPS 이상)**을 동시에 달성하기 위해 CNN(YOLO)과 Transformer(RT-DETR) 등 다양한 SOTA 모델을 분석하고 최적화를 수행하였습니다.

* **기간:** 2025.11.24 ~ 11.26 (3일간의 집중 연구)
* **목표:** mAP 0.8 이상, 20 FPS 이상 (KITTI Dataset)
* **핵심 전략:** 3-Node 분산 학습, 이종 모델 앙상블(Ensemble), 모델 스케일업(Scale-up)

---

## 2. 🔬 실험 과정 및 상세 결과 (Experiment History)

최적의 모델을 선정하기 위해 수행한 **총 8가지의 실험 조건과 결과**입니다.
실패한 실험(ONNX, Augmentation)과 기각된 전략(Ensemble)까지 모두 기록하여 기술적 의사결정의 근거로 삼았습니다.

| 구분 | 모델명 | 적용 전략 (Method) | mAP 50 | mAP 50-95 | FPS | 최종 평가 (Decision) |
| :--: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **YOLOv8s** | **Baseline (SGD)** | **0.9068** | 0.6918 | **536** | **Pass (기준 만족)** |
| 2 | YOLOv8s | Runtime (ONNX) | 0.9096 | 0.6905 | 181 | **Fail** (변환 오버헤드로 속도 저하) |
| 3 | YOLOv8s | Tuning (AdamW) | 0.9028 | 0.6810 | 529 | **Hold** (SGD 대비 이득 미미) |
| 4 | YOLOv8s | Robust (Augmentation) | 0.7086 | 0.4495 | 541 | **Fail** (과도한 왜곡으로 성능 급락) |
| 5 | **YOLOv11m** | **Scale-up (Medium)** | **0.9260** | 0.7210 | **416** | **Good** (밸런스 우수) |
| 6 | **YOLOv11x** | **High-Res (1280px)** | 0.9250 | **0.7420** | 50 | **Good** (박스 정밀도 1위) |
| 7 | **RT-DETR-L** | **Transformer (ViT)** | **0.9330** | **0.7180** | **97** | **🏆 Best (최종 선정)** |
| 8 | Ensemble | WBF (3-Model Mix) | 0.9317 | - | 31 | **Reject** (단일 모델 대비 효율 낮음) |
| 9 | RT-DETR | TTA (Test Time Aug) | 0.9195 | 0.7094 | 39 | **Reject** (위치 정보 교란으로 하락) |

> **💡 결론:** 앙상블(0.9317)이 단일 최고 모델(RT-DETR 0.9330)보다 점수가 낮게 측정됨을 확인했습니다. 이는 성능 격차가 있는 모델들을 평균화(Averaging)하는 과정에서 정밀도가 희석되었기 때문입니다. 이에 따라 **가장 효율적이고 강력한 'RT-DETR-L 단일 모델'을 최종 확정**했습니다.

---

## 3. ⚙️ 기술적 시도 및 해결 전략 (Technical Approach)

### 🔹 3-Node 분산 학습 시스템 (Distributed Training)
시간 단축과 다양한 아키텍처 확보를 위해 3대의 고성능 워크스테이션을 유기적으로 활용했습니다.
* **Node A (Windows):** **YOLOv11x (High-Res 1280px)** 학습 (소형 객체 탐지 강화)
* **Node B (Windows):** **RT-DETR-L (Transformer)** 학습 (전역 문맥 파악)
* **Node C (Mac M4 Max):** **YOLOv9e (GELAN Arch)** 학습 (구조적 다양성 확보)

### 🔹 추론 최적화 검증 (Inference Optimization)
* **WBF (Weighted Box Fusion):** 단순 NMS 대신 다중 모델의 예측값을 가중 평균하여 정밀도 향상을 시도했습니다.
* **TTA (Test Time Augmentation):** 추론 시 이미지를 변환하여 신뢰도를 확보하려 했으나, Transformer 모델에서는 위치 정보 교란 부작용이 있음을 실험적으로 밝혀냈습니다.

---

## 4. 🚀 실행 방법 (How to Run)

### 1. 환경 설정
\\\ash
pip install -r requirements.txt
\\\

### 2. 최종 모델 시각화 (RT-DETR)
최종 선정된 모델로 KITTI 데이터를 추론하고 결과를 시각화합니다.
\\\ash
# 소스 코드 폴더에 있는 시각화 스크립트 실행
python src/final_predict.py
\\\

### 3. 앙상블 실험 재현 (선택 사항)
우리가 수행했던 앙상블 실험을 재현하려면 아래 코드를 실행하세요.
\\\ash
python src/final_ensemble.py
\\\

---

## 5. 💻 개발 환경 (Environment)

* **Hardware:**
    * NVIDIA GeForce RTX 4060 (x2)
    * Apple MacBook Pro M4 Max (128GB Unified Memory)
* **Software:**
    * Python 3.11
    * PyTorch 2.5 (CUDA 12.1 / MPS)
    * Ultralytics YOLO
    * Ensemble-Boxes
"""

# 파일로 저장 (UTF-8 인코딩 필수)
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ README.md 파일 생성 완료!")
