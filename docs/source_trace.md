# Source Trace: UVA DL Tutorial 15 Adaptation

이 문서는 본 과제가 어떤 외부 제공 코드를 출발점으로 삼았고, 무엇을 그대로 사용하지 않고 어떻게 바꾸었는지 추적하기 위한 문서이다.

## 1. Primary Source

- Source: UVA DL Tutorial 15: Vision Transformers
- URL: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
- Assignment role: ViT baseline implementation reference, CIFAR-10 split, patch embedding logic, Pre-LN Transformer block, CLS token, learnable positional embedding

## 2. 가져온 핵심 아이디어

| Source component | Used in this project | Location |
|---|---|---|
| CIFAR-10 train/val/test split | 45k train / 5k validation / 10k test | `src/dataset.py` |
| CIFAR-10 normalization constants | mean/std normalization | `src/dataset.py` |
| training augmentation | RandomHorizontalFlip, RandomResizedCrop | `src/dataset.py` |
| `img_to_patch` idea | image tensor to patch sequence | `src/models.py` |
| Pre-LN Transformer block | LayerNorm before attention/MLP | `src/models.py` |
| linear patch projection | patch vector to `embed_dim` token | `src/models.py` |
| CLS token | default classification token strategy | `src/models.py` |
| learnable positional embedding | position-dependent token information | `src/models.py` |
| AdamW optimizer | baseline optimizer | `src/train.py`, `configs/*.yaml` |

## 3. 그대로 쓰지 않고 바꾼 점

| UVA Tutorial form | This project form | Reason |
|---|---|---|
| Jupyter Notebook | `.py` scripts | 반복 실험, 로그 저장, Git 관리에 유리 |
| PyTorch Lightning | Plain PyTorch loop | 코드 동작을 직접 설명하고 보고서에 연결하기 쉬움 |
| pretrained checkpoint loading | train from scratch | 본인 실험 결과와 plot을 만들어야 함 |
| TensorBoard 중심 | CSV + PNG + JSON 저장 | 리포트 figure와 표로 바로 사용하기 위함 |
| only CLS-token classification | CLS / mean pooling 선택 가능 | student twist 수행을 위해 확장 |

## 4. 보고서에서 사용할 수 있는 설명 방향

본 프로젝트의 ViT 구현은 UVA DL Tutorial 15의 구조를 출발점으로 삼았다. 다만 과제의 목적이 notebook 실행이 아니라 실험 결과를 직접 생성하고 해석하는 것이기 때문에, PyTorch Lightning 기반 notebook 구조를 plain PyTorch `.py` 실험 코드로 재구성하였다. 특히 patch embedding, CLS token, learnable positional embedding, Pre-LN Transformer block은 UVA Tutorial의 핵심 구조를 반영하되, 실험 반복과 비교를 위해 config 기반으로 분리하였다.

## 5. 아직 확인해야 할 것

- 실제 실행 후 `train_log.csv`, `metrics.json`, `learning_curve.png`가 정상 생성되는지 확인
- `pooling_mode=mean`이 CLS token 제거 twist로 정상 동작하는지 확인
- patch size 2/4/8이 token 수, 학습 시간, accuracy에 어떤 차이를 만드는지 확인
