# Run branch note

This branch keeps only the files needed to rerun the project code.

Included:

- `src/`: Python source code
- `configs/`: YAML experiment configs
- `scripts/`: reproduction scripts
- `docs/`: supporting notes
- `data/`: dataset download target placeholder
- `outputs/logs/.gitkeep`, `outputs/results/.gitkeep`: empty output directories for reruns

Removed from this branch:

- generated logs under `outputs/logs/`
- generated experiment outputs under `outputs/results/`
- generated report figures under `report_figures/`
- model checkpoints and weight files
- real CIFAR-10 data files

`main` keeps the lightweight result evidence such as csv, json, log, and png files. `run` is the clean branch for cloning and rerunning the code.

## CIFAR-10 download note

The code uses `torchvision.datasets.CIFAR10` with `data/` as the dataset root. If automatic download is slow, manually download `cifar-10-python.tar.gz` from the official CIFAR-10 page, place it in `data/`, and extract it so that this directory exists:

```text
data/cifar-10-batches-py/
```

Then rerun the same training command.
