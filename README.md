# Vision Transformer Tuning Assignment

## 1. Project Overview

본 프로젝트는 Vision Transformer(ViT)의 구조적 특성과 성능 변화를 CIFAR-10 image classification task에서 실험적으로 분석하기 위한 과제이다.

본 과제의 목표는 단순히 ViT 모델을 실행하는 것이 아니라, 직접 수행한 실험 결과를 바탕으로 다음 항목을 비교하고 해석하는 것이다.

* ViT baseline 학습
* CNN baseline 학습
* patch size 변화에 따른 성능 및 계산 비용 비교
* model capacity 변화에 따른 성능 및 overfitting 비교
* student twist를 적용한 Modified ViT 실험
* learning curve, confusion matrix, wrong prediction examples를 이용한 결과 분석

보고서에서는 모든 해석을 본인이 생성한 figure, plot, table, visualization에 연결하여 작성한다.

---

## 2. Project Directory

Project root:

```bash
/media/cine/First/HWPJ2/2026VAI2
```

예정된 프로젝트 구조는 다음과 같다.

```text
2026VAI2/
├── README.md
├── requirements.txt
├── configs/
│   ├── vit_baseline.yaml
│   ├── cnn_baseline.yaml
│   ├── patch_p2.yaml
│   ├── patch_p4.yaml
│   ├── patch_p8.yaml
│   ├── capacity_small.yaml
│   ├── capacity_large.yaml
│   └── twist_meanpool.yaml
├── src/
│   ├── models.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_results.py
│   └── utils.py
├── data/
└── outputs/
    ├── logs/
    ├── figures/
    ├── checkpoints/
    └── results/
```

---

## 3. Development Environment

본 프로젝트는 PyCharm에서 Python script 방식으로 진행한다.

현재 설정한 Python interpreter는 다음 conda environment를 사용한다.

```bash
/home/cine/anaconda3/envs/vit_tuning/bin/python
```

Conda environment name:

```bash
vit_tuning
```

개발 도구:

```text
PyCharm
```

실행 방식:

```text
.py script 기반 실행
```

GPU 환경은 이후 실행 로그를 통해 확인한 뒤 아래 항목을 보완한다.

```text
Python version: 추후 기록
PyTorch version: 추후 기록
CUDA available: 추후 기록
CUDA version: 추후 기록
GPU: 추후 기록
```

---

## 4. Environment Setup

Conda environment 생성:

```bash
conda create -n vit_tuning python=3.10 -y
```

환경 활성화:

```bash
conda activate vit_tuning
```

만약 `conda activate`가 동작하지 않으면 다음을 먼저 실행한다.

```bash
source /home/cine/anaconda3/etc/profile.d/conda.sh
conda activate vit_tuning
```

PyTorch 설치:

```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

기타 패키지 설치:

```bash
pip install numpy pandas matplotlib scikit-learn tqdm pyyaml pillow torchinfo fvcore
```

---

## 5. Required Packages

주요 패키지와 사용 목적은 다음과 같다.

| Package      | Purpose                                   |
| ------------ | ----------------------------------------- |
| torch        | ViT/CNN model training                    |
| torchvision  | CIFAR-10 dataset loading and transforms   |
| numpy        | numerical computation                     |
| pandas       | experiment result table and csv logging   |
| matplotlib   | learning curve and result plot generation |
| scikit-learn | confusion matrix and evaluation metrics   |
| tqdm         | training progress bar                     |
| pyyaml       | config file loading                       |
| pillow       | image processing                          |
| torchinfo    | parameter count summary                   |
| fvcore       | optional FLOPs calculation                |

---

## 6. Dataset

본 프로젝트는 CIFAR-10 dataset을 사용한다.

CIFAR-10은 10개 class의 32×32 RGB image로 구성된 image classification dataset이다.

사용 예정 class:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

Dataset은 `torchvision.datasets.CIFAR10`을 통해 다운로드하고 로드한다.

예정 저장 위치:

```bash
./data
```

---

## 7. Experiment Plan

본 과제에서 수행할 최소 실험은 다음과 같다.

### 7.1 ViT Baseline

기본 Vision Transformer를 학습한다.

예정 설정:

```text
dataset: CIFAR-10
image size: 32x32
patch size: 4
embedding dimension: 128
depth: 4
attention heads: 4
optimizer: AdamW
loss: CrossEntropyLoss
epoch: 20
```

저장 예정 결과:

```text
train loss curve
train accuracy curve
validation accuracy curve
test accuracy
confusion matrix
wrong prediction examples
training time
parameter count
```

---

### 7.2 CNN Baseline

ViT와 비교하기 위한 CNN baseline을 학습한다.

목적:

```text
CIFAR-10에서 CNN과 ViT의 학습 속도, accuracy, confusion matrix, training time을 비교한다.
```

저장 예정 결과:

```text
CNN learning curve
CNN test accuracy
CNN confusion matrix
CNN wrong prediction examples
CNN parameter count
CNN training time
```

---

### 7.3 Patch Size Comparison

최소 3개 patch size를 비교한다.

예정 조건:

| Experiment | Patch Size | Number of Patch Tokens |
| ---------- | ---------: | ---------------------: |
| ViT-P2     |          2 |                    256 |
| ViT-P4     |          4 |                     64 |
| ViT-P8     |          8 |                     16 |

CLS token을 사용하는 경우 실제 sequence length는 patch token 수에 1을 더한 값이 된다.

저장 예정 결과:

```text
patch size별 test accuracy bar plot
patch size별 training time bar plot
patch size별 token count bar plot
patch size별 learning curve
patch size별 wrong prediction examples
```

---

### 7.4 Model Capacity Comparison

ViT의 model capacity를 바꿔 성능과 overfitting을 비교한다.

예정 조건:

| Experiment | Embedding Dim | Depth | Heads |
| ---------- | ------------: | ----: | ----: |
| ViT-Small  |            64 |     2 |     2 |
| ViT-Base   |           128 |     4 |     4 |
| ViT-Large  |           256 |     6 |     8 |

저장 예정 결과:

```text
parameter count vs accuracy scatter plot
training time vs accuracy scatter plot
train-validation gap bar plot
capacity별 learning curve
```

---

### 7.5 Student Twist

Modified ViT를 1개 이상 설계한다.

우선 적용할 twist 후보:

```text
CLS token 대신 mean pooling 사용
```

비교 조건:

| Experiment   | Classification Token Strategy |
| ------------ | ----------------------------- |
| ViT-CLS      | CLS token output 사용           |
| ViT-MeanPool | patch token 전체 평균 사용          |

저장 예정 결과:

```text
CLS vs MeanPool accuracy comparison
CLS vs MeanPool learning curve
CLS vs MeanPool confusion matrix
CLS vs MeanPool wrong prediction examples
```

---

## 8. How to Run

모든 실험은 project root에서 실행한다.

Project root:

```bash
cd /media/cine/First/HWPJ2/2026VAI2
```

예정 실행 명령어:

```bash
python src/train.py --config configs/vit_baseline.yaml --output_dir outputs/results/vit_baseline
```

CNN baseline:

```bash
python src/train.py --config configs/cnn_baseline.yaml --output_dir outputs/results/cnn_baseline
```

Patch size comparison:

```bash
python src/train.py --config configs/patch_p2.yaml --output_dir outputs/results/patch_p2
python src/train.py --config configs/patch_p4.yaml --output_dir outputs/results/patch_p4
python src/train.py --config configs/patch_p8.yaml --output_dir outputs/results/patch_p8
```

Model capacity comparison:

```bash
python src/train.py --config configs/capacity_small.yaml --output_dir outputs/results/capacity_small
python src/train.py --config configs/capacity_large.yaml --output_dir outputs/results/capacity_large
```

Student twist:

```bash
python src/train.py --config configs/twist_meanpool.yaml --output_dir outputs/results/twist_meanpool
```

---

## 9. Output Files

각 실험은 다음 파일을 저장하도록 구성할 예정이다.

```text
outputs/results/{experiment_name}/
├── config.yaml
├── train_log.csv
├── metrics.json
├── checkpoint.pt
├── learning_curve.png
├── confusion_matrix.png
├── wrong_examples.png
└── summary.txt
```

통합 결과 파일:

```text
outputs/results/summary_results.csv
```

보고서용 figure 저장 위치:

```text
outputs/figures/
```

---

## 10. Report Mapping

보고서에는 다음 figure를 포함할 예정이다.

| Figure    | Content                               |
| --------- | ------------------------------------- |
| Figure 1  | CIFAR-10 dataset sample visualization |
| Figure 2  | Patch split visualization             |
| Figure 3  | ViT baseline learning curve           |
| Figure 4  | CNN baseline learning curve           |
| Figure 5  | ViT vs CNN accuracy comparison        |
| Figure 6  | Patch size별 accuracy comparison       |
| Figure 7  | Parameter count vs accuracy           |
| Figure 8  | Confusion matrix                      |
| Figure 9  | Wrong prediction examples             |
| Figure 10 | Student twist result plot             |

---

## 11. Reproducibility Notes

실험 재현을 위해 다음 정보를 기록한다.

```text
random seed
batch size
epoch
learning rate
optimizer
model hyperparameters
training time
parameter count
GPU information
```

각 실험은 config file로 hyperparameter를 분리하여 실행하고, 결과 파일은 experiment name별로 저장한다.

---

## 12. Current Status

현재 완료된 작업:

```text
project root 설정 완료
PyCharm project 설정 완료
conda environment vit_tuning 생성 완료
PyCharm interpreter 연결 완료
필수 package 설치 완료
README 초안 작성 중
```

아직 완료되지 않은 작업:

```text
GPU/CUDA 실행 로그 저장
src 코드 작성
config 파일 작성
dataset loading 테스트
ViT baseline 학습
CNN baseline 학습
patch size 비교 실험
model capacity 비교 실험
student twist 실험
보고서 작성
```
