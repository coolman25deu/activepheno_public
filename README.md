# ActivePheno

ActivePheno is a reproducible framework for multimodal live-cell phenotypic profiling using microscopy-derived and molecular representations.

The public pipeline supports live-cell microscopy, frozen DINOv3 image representations, molecular representations including Morgan fingerprints and MolFormer, measured-timepoint MoA classification, and compound-disjoint or scaffold-disjoint evaluation. Data preparation, feature extraction, model selection, and evaluation are configurable and reproducible.

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,chemistry,baselines]'
```

For image extraction, install the `images` extra with a PyTorch build suited to your device. MolFormer support is available through the `molformer` extra. DINOv3 weights may require approved Hugging Face access; configure credentials outside this repository.

## Data and example workflow

The [LCTP dataset](https://doi.org/10.17044/SCILIFELAB.29315642) is distributed by its authors under CC BY-NC-SA 4.0. This repository does not redistribute images. The sample downloader retrieves public metadata and three example images for ingestion checks:

```bash
python scripts/download_lctp_sample.py --output-root data/raw/lctp
python scripts/prepare_lctp.py --config configs/data/lctp.yaml
```

The sample is too small for model fitting or split generation. For full data, follow the upstream data access instructions, provide the metadata and plate-start table, then run preparation with a local configuration. See [examples/README.md](examples/README.md) for the full pipeline sequence and input contracts.

Compound-disjoint splits keep every compound and its trajectories in one partition. Scaffold-disjoint splits additionally keep Bemis–Murcko scaffolds together. Both methods validate compound identity, labels, structures, and trajectory integrity before writing a manifest.

Image observations are measured frames only. Time is elapsed from the first scan unless the source provides a verified treatment-time origin. Evaluation uses acquired image frames at documented elapsed times.

## Reproducibility

Configurations pin DINO checkpoint revisions and keep data, cache, and output paths configurable. Splits and model selection use explicit seeds; hyperparameters are selected on validation compounds. Tests use small synthetic fixtures and do not download pretrained weights.

This repository contains the reproducible core pipeline and selected demonstration results. Additional benchmarking and robustness analyses are being prepared for a manuscript.
