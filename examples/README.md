# Reproduction workflow

Run commands from the repository root. The sample includes one compound and is only for validating ingestion and image feature extraction; split generation and MoA evaluation require a full, locally prepared dataset.

1. Install the chemistry and image extras, then download metadata and the small image sample with `python scripts/download_lctp_sample.py`.
2. Prepare observations with `python scripts/prepare_lctp.py --config configs/data/lctp.yaml`.
3. Resolve structures from the upstream LCTP supplementary table using `python scripts/resolve_compounds.py --metadata data/raw/lctp/metadata.parquet.gzip --catalog data/raw/lctp/Supplementary_Table_6.html --output-root data/processed/chemistry`.
4. For full data, extract cached image representations with `python scripts/extract_image_embeddings.py --config configs/features/dinov3.yaml --observations data/processed/lctp/observations.parquet --raw-root data/raw/lctp --output-root data/cache/dinov3 --device cuda`.
5. Generate a compound- or scaffold-disjoint split with `python scripts/make_splits.py --help`, then run the configured measured-timepoint example with `python scripts/train_moa.py --config configs/example.yaml`. Edit the paths and measured hour in that YAML for your local data.

The pipeline does not bundle images, compound tables, embeddings, split manifests, or trained models. Generated files stay in user-selected local directories.
