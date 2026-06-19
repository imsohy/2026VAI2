# Source Trace: UVA DL Tutorial 15 Adaptation

이 문서는 본 과제가 어떤 외부 제공 코드를 출발점으로 삼았고, 무엇을 그대로 유지했으며, 무엇을 과제 실험 목적에 맞게 바꾸었는지 추적하기 위한 문서이다.

## 1. Primary Source

- Source: UVA DL Tutorial 15: Vision Transformers
- URL: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
- Assignment role: ViT baseline implementation reference, CIFAR-10 split, patch embedding logic, Pre-LN Transformer block, CLS token, learnable positional embedding, PyTorch Lightning training structure

## 2. 가져온 핵심 아이디어

| Source component | Used in this project | Location |
|---|---|---|
| CIFAR-10 train/val/test split | 45k train / 5k validation / 10k test | `src/dataset.py` |
| CIFAR-10 normalization constants | mean/std normalization | `src/dataset.py` |
| training augmentation | RandomHorizontalFlip, RandomResizedCrop | `src/dataset.py` |
| `img_to_patch` idea | image tensor to patch sequence | `src/models.py` |
| linear patch projection | patch vector to `embed_dim` token | `src/models.py` |
| Pre-LN Transformer block | LayerNorm before attention/MLP | `src/models.py` |
| CLS token | default classification token strategy | `src/models.py` |
| learnable positional embedding | position-dependent token information | `src/models.py` |
| PyTorch Lightning training style | LightningModule + Trainer 기반 학습 관리 | `src/lit_module.py`, `src/train.py` |
| AdamW optimizer | baseline optimizer | `src/lit_module.py`, `configs/*.yaml` |
| MultiStepLR scheduler | UVA 원본의 장기 학습 scheduler로 기록만 유지 | 현재 20 epoch 실험에서는 사용하지 않음 |

## 3. UVA 원본 baseline과 본 과제 baseline의 관계

`configs/vit_baseline.yaml`은 UVA Tutorial 15의 ViT architecture hyperparameter를 기준으로 맞춘다. 다만 학습 epoch는 과제 요구사항의 최소 기준과 전체 실험량을 고려하여 20으로 통일한다.

| Item | UVA Tutorial 15 value | This project baseline |
|---|---:|---:|
| patch_size | 4 | 4 |
| num_patches | 64 | model에서 자동 계산 |
| embed_dim | 256 | 256 |
| hidden_dim | 512 | 512 |
| num_heads | 8 | 8 |
| num_layers | 6 | 6 |
| dropout | 0.2 | 0.2 |
| learning_rate | 3e-4 | 3e-4 |
| batch_size | 128 | 128 |
| epochs | 180 | 20 |
| optimizer | AdamW | AdamW |
| scheduler | MultiStepLR | none |
| lr milestones | [100, 150] | [] |
| lr gamma | 0.1 | 0.1, unused |
| train/val/test split | 45k / 5k / 10k | 45k / 5k / 10k |

주의: `configs/debug_vit.yaml`은 빠른 smoke test용이므로 UVA baseline과 다르다. 실제 보고서의 baseline 기준은 `configs/vit_baseline.yaml`이다.

## 4. 그대로 쓰지 않고 바꾼 점

| UVA Tutorial form | This project form | Reason |
|---|---|---|
| Jupyter Notebook | `.py` scripts | 반복 실험, 로그 저장, Git 관리, 과제 제출에 유리 |
| PyTorch Lightning | PyTorch Lightning 유지 | 교수님 제공 reference의 학습 framework를 유지하여 출발점 명확화 |
| tutorial 내부 고정 설정 | YAML config 기반 설정 | patch size, model capacity, pooling strategy를 실험별로 바꾸기 위함 |
| 180 epoch 장기 학습 budget | 20 epoch 통일 실험 budget | 과제 최소 조건을 만족하면서 전체 필수 실험을 같은 조건에서 비교하기 위함 |
| MultiStepLR milestones [100,150] | scheduler none | 20 epoch에서는 100/150 milestone이 작동하지 않으므로 혼동 방지 |
| pretrained checkpoint loading 가능 | train from scratch 중심 | 본인 실험 결과와 plot을 만들어야 함 |
| TensorBoard 중심 | CSV + PNG + JSON도 함께 저장 | 리포트 figure와 표로 바로 사용하기 위함 |
| only CLS-token classification | CLS / mean pooling 선택 가능 | student twist 수행을 위해 확장 |
| 단일 tutorial 실행 흐름 | ViT/CNN/patch/capacity/twist 실험 스크립트 분리 | 과제 필수 실험을 빠짐없이 수행하기 위함 |

## 5. 보고서에서 사용할 수 있는 설명 방향

본 프로젝트의 ViT baseline은 UVA DL Tutorial 15의 CIFAR-10 Vision Transformer 예제를 출발점으로 삼았다. 원 tutorial의 PyTorch Lightning 기반 학습 구조, CIFAR-10 data preparation, image-to-patch 변환, linear patch projection, CLS token, learnable positional embedding, Pre-LN Transformer block 구조를 유지하였다. 또한 `configs/vit_baseline.yaml`의 patch size, embedding dimension, hidden dimension, attention head 수, transformer depth, dropout, learning rate, batch size는 UVA Tutorial 15의 baseline 설정과 동일하게 맞추었다.

다만 본 과제의 목적은 단순히 tutorial의 장기 학습 결과를 재현하는 것이 아니라, 동일한 조건 아래에서 patch size, model capacity, pooling strategy가 결과에 미치는 상대적 차이를 분석하는 것이다. 따라서 UVA의 180 epoch 학습 budget은 원본 reference로 기록하되, 실제 모든 실험은 과제 최소 조건인 20 epoch로 통일하였다. 20 epoch 실험에서는 UVA의 `[100, 150]` learning rate milestone이 작동하지 않으므로 scheduler는 사용하지 않았다.

이를 위해 notebook 구조를 `.py` script와 YAML config 구조로 재구성하였다. 각 실험은 `train_log.csv`, `metrics.json`, learning curve, confusion matrix, wrong prediction examples, parameter count, training time을 저장하도록 확장하였다.

## 6. 교수님 강의/과제 의도와의 연결

교수님은 ViT에서 이미지를 patch 단위로 나누고 각 patch를 token처럼 처리한다고 설명하였다. 또한 patch token을 linear projection으로 model dimension에 맞추고, positional encoding과 CLS token을 더해 Transformer encoder에 넣는 구조를 강조하였다. 이 프로젝트의 `models.py`는 이 흐름을 코드로 구현한다.

교수님은 동시에 model dimension, token 수, patch 수, attention head, CNN feature 사용 여부 같은 설계 선택을 학생이 전략적으로 바꿔볼 수 있다고 설명하였다. 따라서 본 프로젝트는 Lightning framework 자체를 바꾸는 것보다, config를 통해 구조적 실험 변수를 바꾸고 그 결과를 figure로 비교하는 것을 핵심으로 둔다.

## 7. 아직 확인해야 할 것

- 실제 20 epoch baseline 실행 후 `train_log.csv`, `metrics.json`, `learning_curve.png`가 정상 생성되는지 확인
- 20 epoch budget에서 ViT baseline이 어느 시점부터 수렴/정체하는지 확인
- `pooling_mode=mean`이 CLS token 대신 mean pooling을 사용하는 twist로 정상 동작하는지 확인
- patch size 2/4/8이 token 수, 학습 시간, accuracy에 어떤 차이를 만드는지 확인
- capacity small/large가 parameter count, train-validation gap, accuracy에 어떤 차이를 만드는지 확인
- CNN baseline과 ViT baseline의 수렴 속도, confusion matrix, training time 차이를 확인
