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
| basic training augmentation | RandomHorizontalFlip, RandomResizedCrop 또는 RandomCrop | `src/dataset.py` |
| `img_to_patch` idea | image tensor to patch sequence | `src/models.py` |
| linear patch projection | patch vector to `embed_dim` token | `src/models.py` |
| Pre-LN Transformer block | LayerNorm before attention/MLP | `src/models.py` |
| CLS token | default classification token strategy | `src/models.py` |
| learnable positional embedding | position-dependent token information | `src/models.py` |
| PyTorch Lightning training style | LightningModule + Trainer 기반 학습 관리 | `src/lit_module.py`, `src/train.py` |
| AdamW optimizer | baseline optimizer | `src/lit_module.py`, `configs/*.yaml` |
| MultiStepLR scheduler | UVA 원본의 180 epoch 장기 학습 scheduler | 본 최종 실험에서는 `scheduler: none` |

## 3. UVA 원본 baseline과 본 과제 baseline의 관계

`configs/vit_baseline_e100.yaml`은 UVA Tutorial 15의 ViT architecture hyperparameter를 기준으로 맞춘다. 다만 원본 tutorial의 180 epoch 장기 학습을 그대로 재현하는 대신, 본 과제에서는 필수 실험을 모두 수행하고 동일한 budget으로 비교하기 위해 최종 실험을 100 epoch로 통일하였다. 20 epoch config들은 예비 실행 및 요구 최소 기준 확인용으로 남겼다.

| Item | UVA Tutorial 15 value | This project final baseline |
|---|---:|---:|
| dataset | CIFAR-10 | CIFAR-10 |
| image size | 32×32 | 32×32 |
| patch_size | 4 | 4 |
| num_patches | 64 | model에서 자동 계산, 64 |
| embed_dim | 256 | 256 |
| hidden_dim | 512 | 512 |
| num_heads | 8 | 8 |
| num_layers | 6 | 6 |
| dropout | 0.2 | 0.2 |
| learning_rate | 3e-4 | 3e-4 |
| batch_size | 128 | 128 |
| epochs | 180 | 100 |
| optimizer | AdamW | AdamW |
| scheduler | MultiStepLR [100, 150] | none |
| train/val/test split | 45k / 5k / 10k | 45k / 5k / 10k |

주의: `configs/debug_vit.yaml`은 빠른 smoke test용이고, `configs/*_e100.yaml`이 최종 보고서 기준 실험이다.

## 4. 그대로 쓰지 않고 바꾼 점

| UVA Tutorial form | This project form | Reason |
|---|---|---|
| Jupyter Notebook | `.py` scripts | 반복 실험, 로그 저장, Git 관리, 과제 제출에 유리 |
| tutorial 내부 고정 설정 | YAML config 기반 설정 | patch size, model capacity, augmentation을 실험별로 바꾸기 위함 |
| 180 epoch 장기 학습 budget | 100 epoch final comparison | 과제의 필수 실험을 현실적인 시간 안에서 모두 비교하기 위함 |
| MultiStepLR milestones [100,150] | scheduler none | 100 epoch에서는 milestone 100이 마지막 지점이라 비교 해석이 애매하고, 모든 실험을 동일한 기본 학습 규칙으로 맞추기 위함 |
| pretrained checkpoint loading 가능 | train from scratch 중심 | 본인 실험 결과와 plot을 만들어야 함 |
| TensorBoard 중심 | CSV + PNG + JSON + summary 저장 | 리포트 figure와 표로 바로 사용하기 위함 |
| only CLS-token classification | pooling mode 옵션 추가 | mean pooling twist를 시도하기 위함. 최종 보고서에서는 제외 |
| tutorial 기본 augmentation | basic / AutoAugment 분기 | overfitting 완화 실험을 위해 `augmentation_policy` 추가 |

## 5. AutoAugment twist의 출처와 위치

최종 Modified ViT는 모델 구조를 키우지 않고, 학습 데이터 변형 정책을 강화하는 방식으로 설계하였다.

- Inspiration: DACON image classification notebook에서 사용된 AutoAugment 계열 아이디어
- Implementation in this project: `torchvision.transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)`
- Code location: `src/dataset.py`
- Config: `configs/twist_autoaugment_e100.yaml`
- Output: `outputs/results/twist_autoaugment_e100`

AutoAugment는 baseline ViT와 동일한 patch size, embedding dimension, transformer depth, attention head 수, optimizer, epoch를 유지한 상태에서 training transform만 바꾼다. 따라서 성능 변화는 model capacity 증가가 아니라 augmentation/regularization 전략 변화로 해석할 수 있다.

## 6. 보고서에서 사용할 수 있는 설명 방향

본 프로젝트의 ViT baseline은 UVA DL Tutorial 15의 CIFAR-10 Vision Transformer 예제를 출발점으로 삼았다. 원 tutorial의 PyTorch Lightning 기반 학습 구조, CIFAR-10 data preparation, image-to-patch 변환, linear patch projection, CLS token, learnable positional embedding, Pre-LN Transformer block 구조를 유지하였다.

다만 본 과제의 목적은 tutorial 결과를 그대로 복제하는 것이 아니라, 동일한 조건에서 patch size, model capacity, CNN baseline, Modified ViT가 결과에 미치는 상대적 차이를 본인 실험 plot으로 설명하는 것이다. 따라서 notebook 구조를 `.py` script와 YAML config 구조로 재구성하고, 각 실험이 `train_log.csv`, `metrics.json`, learning curve, confusion matrix, wrong prediction examples, parameter count, training time을 저장하도록 확장하였다.

최종 보고서에서는 100 epoch 결과를 기준으로 한다. 20 epoch 결과는 예비 실험으로 남겨두되, 최종 수치와 figure는 `*_e100` 결과를 사용한다.

## 7. 교수님 강의/과제 의도와의 연결

교수님은 ViT에서 이미지를 patch 단위로 나누고 각 patch를 token처럼 처리한다고 설명하였다. 또한 patch token을 linear projection으로 model dimension에 맞추고, positional encoding과 CLS token을 더해 Transformer encoder에 넣는 구조를 강조하였다. 이 프로젝트의 `models.py`는 이 흐름을 코드로 구현한다.

교수님은 동시에 model dimension, token 수, patch 수, attention head, CNN feature 사용 여부 같은 설계 선택을 학생이 전략적으로 바꿔볼 수 있다고 설명하였다. 따라서 본 프로젝트는 Lightning framework 자체를 바꾸는 것보다, config를 통해 구조적 실험 변수를 바꾸고 그 결과를 figure로 비교하는 것을 핵심으로 둔다.

또한 교수님은 ViT가 데이터가 많고 복잡한 경우에 강점을 보이지만, 단순한 small dataset에서는 CNN으로도 상당 부분 커버될 수 있다고 설명하였다. 본 실험에서 baseline ViT가 100 epoch에서 overfitting 경향을 보인 점, 그리고 AutoAugment가 같은 parameter 수에서 test accuracy를 개선한 점은 이 논의와 연결된다.

## 8. 최종 보고서에서 제외한 것

- `twist_meanpool_e100`: CLS token 대신 mean pooling을 사용하는 시도였으나, AutoAugment가 더 명확한 성능 개선을 보였으므로 final Modified ViT figure에서는 제외하였다.
- `debug_vit`: smoke test용 결과이므로 보고서 수치로 사용하지 않는다.
- checkpoint files: 용량 문제로 Git에 올리지 않으며, 보고서 근거는 csv/json/png/log 파일로 충분하다.
