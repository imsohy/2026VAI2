# Vision Transformer Tuning Assignment

## 1. Project Overview

본 프로젝트는 CIFAR-10 image classification에서 Vision Transformer(ViT)의 구조적 특성과 성능 변화를 실험적으로 분석하기 위한 과제이다.

과제의 핵심은 직접 학습한 결과에서 나온 figure, plot, table을 근거로 다음 질문에 답하는 것이다.

- ViT는 이미지를 patch sequence로 바꾸어 어떻게 분류하는가?
- patch size가 token 수, 학습 시간, 성능에 어떤 영향을 주는가?
- model capacity를 키우면 성능이 항상 좋아지는가?
- CNN baseline과 ViT baseline은 CIFAR-10에서 어떤 차이를 보이는가?
- small dataset에서 ViT가 overfitting되는 경우, augmentation 기반 twist가 일반화 성능을 개선할 수 있는가?

최종 보고서의 주력 Modified ViT는 **AutoAugment를 적용한 ViT**이다. Mean pooling twist는 실험 기록으로는 남겼지만 최종 report figure에서는 제외하였다.

---

## 2. Project Directory

```text
2026VAI2/
├── README.md                         # 프로젝트 개요, 실행 방법, 제출용 정리
├── configs/                          # 실험별 YAML 설정
│   ├── vit_baseline.yaml             # 20 epoch 예비 ViT baseline
│   ├── cnn_baseline.yaml             # 20 epoch 예비 CNN baseline
│   ├── patch_p2.yaml                 # 20 epoch patch size 2
│   ├── patch_p4.yaml                 # 20 epoch patch size 4, baseline과 동일
│   ├── patch_p8.yaml                 # 20 epoch patch size 8
│   ├── capacity_small.yaml           # 20 epoch small ViT
│   ├── capacity_large.yaml           # 20 epoch large ViT
│   ├── twist_meanpool.yaml           # 예비 mean pooling twist, 최종 보고서 제외
│   ├── vit_baseline_e100.yaml        # 최종 ViT baseline, patch size 4 결과로도 사용
│   ├── cnn_baseline_e100.yaml        # 최종 CNN baseline
│   ├── patch_p2_e100.yaml            # 최종 patch size 2
│   ├── patch_p4_e100.yaml            # baseline과 동일하므로 보통 별도 실행하지 않음
│   ├── patch_p8_e100.yaml            # 최종 patch size 8
│   ├── capacity_small_e100.yaml      # 최종 small ViT
│   ├── capacity_large_e100.yaml      # 최종 large ViT
│   ├── twist_meanpool_e100.yaml      # 실패/보조 twist 기록용, 최종 figure 제외
│   └── twist_autoaugment_e100.yaml   # 최종 Modified ViT, AutoAugment 적용
├── src/                              # 실제 학습, 모델, 데이터셋, 평가 코드
│   ├── dataset.py                    # CIFAR-10 로딩, basic/AutoAugment transform
│   ├── models.py                     # ViT, CNN, patch embedding, pooling mode
│   ├── lit_module.py                 # PyTorch LightningModule
│   ├── train.py                      # 학습 entry point, csv/json/png 결과 저장
│   ├── evaluate.py                   # test evaluation, class-wise 결과
│   ├── plot_results.py               # learning curve, confusion matrix, wrong examples
│   └── utils.py                      # config, seed, json, summary helper
├── scripts/                          # 재현용 실행 스크립트와 보고서 figure 생성기
│   ├── build_report_figures.py       # outputs/results를 읽어 report_figures 생성
│   ├── run_baselines.sh              # e100 CNN/ViT baseline 재현용
│   ├── run_patch_compare.sh          # e100 patch size 실험 재현용
│   ├── run_capacity_compare.sh       # e100 capacity 실험 재현용
│   ├── run_autoaugment.sh            # AutoAugment 20/e100 재현용
│   └── run_twist.sh                  # 최종 twist, AutoAugment 중심 재현용
├── docs/                             # 보고서 작성을 위한 추적 문서
│   ├── source_trace.md               # UVA 원본과 우리 코드의 대응 관계
│   ├── figure_inventory.md           # 보고서 figure 번호와 파일 매핑
│   └── experiment_log_template.md    # 실험 기록 템플릿
├── outputs/                          # 실험 결과, 큰 checkpoint는 gitignore
│   ├── logs/                         # tee로 저장한 실행 로그
│   └── results/                      # 실험별 metrics/train_log/figure/summary
├── report_figures/                   # 최종 보고서에 넣을 통합 figure
└── data/                             # CIFAR-10 다운로드 위치, 실제 데이터는 gitignore
```

---

## 3. Environment Setup

본 프로젝트는 PyCharm에서 `.py` script 방식으로 수행하였다.

```bash
cd /media/cine/First/HWPJ2/2026VAI2
```

Conda environment:

```bash
conda create -n vit_tuning python=3.10 -y
source /home/cine/anaconda3/etc/profile.d/conda.sh
conda activate vit_tuning
```

PyTorch 설치:

```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

기타 패키지:

```bash
pip install numpy pandas matplotlib scikit-learn tqdm pyyaml pillow torchinfo fvcore pytorch-lightning tensorboard
```

실험 환경:

```text
Python: 3.10
Framework: PyTorch + PyTorch Lightning
GPU: NVIDIA RTX A6000
Dataset: CIFAR-10
Seed: 42
```

---

## 4. Dataset

- Dataset: CIFAR-10
- Image size: 32×32 RGB
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Split: 45,000 train / 5,000 validation / 10,000 test
- Data root: `./data`

`torchvision.datasets.CIFAR10`으로 다운로드하며, `data/` 내부 실제 데이터 파일은 repository에 올리지 않는다.

---

## 5. Final Experiment Set

최종 보고서는 100 epoch 결과를 중심으로 작성되었다. 20 epoch 결과는 예비 실행 및 sanity check 성격으로 사용하였다.

| Experiment | Role | Main setting | Output directory |
|---|---|---|---|
| `cnn_baseline_e100` | CNN baseline | CNN, 100 epoch | `outputs/results/cnn_baseline_e100` |
| `vit_baseline_e100` | ViT baseline / patch p4 | patch=4, embed=256, depth=6, heads=8 | `outputs/results/vit_baseline_e100` |
| `patch_p2_e100` | Patch comparison | patch=2, token=256 | `outputs/results/patch_p2_e100` |
| `patch_p8_e100` | Patch comparison | patch=8, token=16 | `outputs/results/patch_p8_e100` |
| `capacity_small_e100` | Capacity comparison | smaller embed/depth/head | `outputs/results/capacity_small_e100` |
| `capacity_large_e100` | Capacity comparison | larger embed/depth/head | `outputs/results/capacity_large_e100` |
| `twist_autoaugment_e100` | Final Modified ViT | same ViT + AutoAugment | `outputs/results/twist_autoaugment_e100` |

`patch_p4_e100`은 `vit_baseline_e100`과 같은 조건이므로 최종 patch size 4 결과는 baseline 결과를 재사용한다.

---

## 6. How We Ran the Experiments

실제 실험은 PyCharm terminal에서 각 config를 직접 실행하고, console 출력과 로그를 동시에 남기기 위해 `2>&1 | tee ...` 형식을 사용하였다.

기본 형식:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONUNBUFFERED=1 python -u src/train.py \
  --config <CONFIG_PATH> \
  --output_dir <OUTPUT_DIR> \
  2>&1 | tee <LOG_PATH>
```

예: 최종 ViT baseline

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/vit_baseline_e100.yaml \
  --output_dir outputs/results/vit_baseline_e100 \
  2>&1 | tee outputs/logs/vit_baseline_e100_run.log
```

예: 최종 AutoAugment twist

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u src/train.py \
  --config configs/twist_autoaugment_e100.yaml \
  --output_dir outputs/results/twist_autoaugment_e100 \
  2>&1 | tee outputs/logs/twist_autoaugment_e100_run.log
```

---

## 7. Optional Reproduction Scripts

`scripts/` 폴더의 shell script는 실제 실험을 다시 재현하기 위한 convenience wrapper이다. 본 실험은 주로 위의 개별 명령으로 실행했지만, 동일한 config/output/log 규칙을 재사용할 수 있도록 script를 남겨두었다.

```bash
# 최종 CNN/ViT baseline 재현
bash scripts/run_baselines.sh

# 최종 patch size 비교 재현
bash scripts/run_patch_compare.sh

# 최종 model capacity 비교 재현
bash scripts/run_capacity_compare.sh

# 최종 AutoAugment twist 재현
bash scripts/run_twist.sh

# 저장된 outputs/results를 이용해 통합 report figure 생성
python scripts/build_report_figures.py
```

`build_report_figures.py`는 모델을 다시 학습하지 않는다. 이미 생성된 `outputs/results/<experiment>/metrics.json`과 `train_log.csv`만 읽어 `report_figures/`에 통합 figure를 만든다.

---

## 8. Output Files

각 실험은 다음 구조로 저장된다.

```text
outputs/results/<experiment>/
├── config.yaml             # 실행 당시 config 복사본
├── train_log.csv           # epoch별 train_loss/train_acc/val_loss/val_acc
├── metrics.json            # best_val_acc, test_acc, test_loss, parameter count, time
├── summary.txt             # 사람이 읽기 쉬운 요약
└── figures/
    ├── learning_curve.png
    ├── learning_curve_loss.png
    ├── confusion_matrix.png
    ├── wrong_examples.png
    └── patch_visualization.png 또는 dataset sample 관련 figure
```

통합 보고서 figure는 다음 위치에 저장된다.

```text
report_figures/
├── summary_metrics.csv
├── model_test_accuracy.png
├── cnn_vs_vit_accuracy.png
├── training_time_comparison.png
├── patch_accuracy.png
├── patch_training_time.png
├── patch_token_count.png
├── param_vs_accuracy.png
├── time_vs_accuracy.png
├── train_val_gap.png
├── vit_baseline_overfit_curve.png
├── autoaugment_overfit_curve.png
├── cnn_vit_val_curve.png
├── patch_val_curve.png
├── capacity_val_curve.png
├── twist_accuracy.png
├── twist_val_curve.png
├── autoaugment_effect.png
└── autoaugment_test_loss.png
```

---

## 9. Final Result Summary

| Experiment | Test Acc | Best Val Acc | Params | Training Time |
|---|---:|---:|---:|---:|
| `cnn_baseline_e100` | 0.8663 | 0.8706 | 559,178 | 13m 11s |
| `vit_baseline_e100` | 0.7584 | 0.7614 | 3,195,146 | 31m 42s |
| `twist_autoaugment_e100` | 0.8209 | 0.8248 | 3,195,146 | 38m 21s |
| `patch_p2_e100` | 0.7473 | 0.7598 | 3,235,082 | 1h 34m 49s |
| `patch_p8_e100` | 0.7230 | 0.7324 | 3,219,722 | 33m 38s |
| `capacity_small_e100` | 0.7280 | 0.7346 | 413,706 | 20m 47s |
| `capacity_large_e100` | 0.7556 | 0.7692 | 9,519,754 | 45m 9s |

핵심 관찰:

- CNN baseline은 ViT baseline보다 더 높은 test accuracy를 보였다.
- ViT baseline은 100 epoch에서 train accuracy가 높아졌지만 validation/test 성능은 상대적으로 정체되어 overfitting 경향을 보였다.
- AutoAugment를 적용한 ViT는 같은 parameter 수에서 test accuracy를 0.7584에서 0.8209로 개선하였다.
- Patch size 2는 token 수가 많아 계산 시간이 크게 증가했지만, 성능 향상은 제한적이었다.
- Large ViT는 parameter 수가 크게 증가했지만, baseline 대비 성능 향상이 크지 않았다.

---

## 10. Report Mapping

보고서에 figure, plot, table은 다음과 같이 사용된다.

| Report Figure | File | Purpose |
|---|---|---|
| Figure 1 | `outputs/results/vit_baseline_e100/figures/dataset_samples.png` 또는 수동 배치 이미지 | CIFAR-10 sample 확인 |
| Figure 2 | `outputs/results/vit_baseline_e100/figures/patch_visualization.png` | ViT patch split 설명 |
| Figure 3 | `outputs/results/vit_baseline_e100/figures/learning_curve.png` | ViT baseline 학습 곡선 |
| Figure 4 | `outputs/results/cnn_baseline_e100/figures/learning_curve.png` | CNN baseline 학습 곡선 |
| Figure 5 | `report_figures/cnn_vs_vit_accuracy.png` | CNN vs ViT vs AutoAugment 성능 비교 |
| Figure 6 | `report_figures/patch_accuracy.png`, `patch_training_time.png`, `patch_token_count.png` | patch size별 성능/비용/token 수 비교 |
| Figure 7 | `report_figures/param_vs_accuracy.png`, `time_vs_accuracy.png` | model capacity와 비용 대비 성능 |
| Figure 8 | 각 실험의 `figures/confusion_matrix.png` | class별 오류 패턴 |
| Figure 9 | 각 실험의 `figures/wrong_examples.png` | 오분류 예시 분석 |
| Figure 10 | `report_figures/autoaugment_effect.png`, `twist_accuracy.png`, `autoaugment_overfit_curve.png` | AutoAugment twist 결과 |

---

## 11. Reproducibility Notes

- 모든 최종 실험은 CIFAR-10, seed 42, batch size 128, AdamW, learning rate 3e-4 조건을 기본으로 한다.
- 최종 비교는 100 epoch budget 기준으로 수행한다.
- `outputs/results/*/config.yaml`은 실행 당시 사용한 config를 보존한다.
- checkpoint 파일은 용량 문제로 Git에 올리지 않는다.
- `metrics.json`, `train_log.csv`, `summary.txt`, `figures/*.png`, `outputs/logs/*.log`, `report_figures/*.png`는 보고서 근거 자료로 사용한다.

---

## 12. Notes on Failed / Auxiliary Experiments

`twist_meanpool_e100`은 CLS token 대신 mean pooling을 사용하는 보조 twist 실험이다. 그러나 최종 Modified ViT로는 AutoAugment가 더 명확한 성능 개선을 보였으므로, mean pooling은 최종 report figure에서 제외하였다.

Mean pooling 관련 config와 raw output은 실험 기록 차원에서 남겨둘 수 있지만, 최종 보고서의 핵심 twist는 `twist_autoaugment_e100`이다.
