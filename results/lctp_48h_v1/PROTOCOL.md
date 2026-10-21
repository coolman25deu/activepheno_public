# Selected LCTP 48 h example

This is a descriptive aggregate excerpt from an existing frozen-representation
benchmark. The endpoint and four methods are shown consistently across both
split types and both input conditions. It is not a best-time or best-seed search.
These test sets were inspected in earlier research; this release is exploratory.

## Data and evaluation unit

The matched cohort contains 72 compounds in six labelled classes (AKT, CDK,
HDAC, MAPK, PARP, TUB), 1,784 well/site trajectories, and five plates: three
A549 and two U2OS plates. Hours are relative to first image acquisition, not a
verified treatment-addition time. Only measured images enter the inputs.

- **Single time:** the real image acquired at 48 h only.
- **Cumulative history:** an equal temporal mean of real frame embeddings at
  0, 2, ..., 48 h (25 scans), before classification. Morgan is static and is
  identical in both conditions.
- **Compound-disjoint:** each compound stays in one partition. There are 42
  training, 18 validation and 12 test compounds per seed.
- **Scaffold-disjoint:** Bemis–Murcko scaffold groups stay in one partition.
  These splits have 43–45 training, 13–16 validation and 13–14 test compounds.
- Seeds are **42, 43 and 44**, unchanged from the original fits. Each reported
  row equally averages all three seeds; no seed is omitted.

The cohort requires complete matched image and author-feature availability on
the 0:2:68 h grid, including for this earlier endpoint. No future feature values
enter a model, but later availability may induce selection bias. This is the
trajectory-level matched protocol; older compound-averaged analyses are not
interchangeable with it.

## Representations and selection

DINOv2 and DINOv3 are frozen ViT-B image encoders. The 1040 × 1408 grayscale
frames are mapped to RGB using a fixed 0–255 scale, with no contrast stretching.
All nonoverlapping native 224 × 224 tiles are retained, with replicated right
and bottom edges: 35 tiles per frame. Tiles receive ImageNet normalization.
Patch tokens are averaged, excluding special/register tokens; tile embeddings
are equally averaged to a 768-dimensional frame embedding. Extraction used BF16.
There is no segmentation or cell selection in this image path.

| Encoder | Checkpoint | Revision |
|---|---|---|
| DINOv2 | `facebook/dinov2-base` | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bbdb9737e363e781fb0c67687896bc` |

Morgan uses canonical SMILES, radius 2 and 2,048 binary bits. DINOv3 + Morgan
concatenates the image and fingerprint blocks. It is a specific standardized
early-fusion baseline, not a claim about the best possible fusion method.

All models use multinomial logistic regression with `lbfgs`, 3,000 maximum
iterations, and C in {0.01, 0.1, 1, 10}. Image and fusion inputs use a
training-fitted weighted StandardScaler; Morgan-only inputs are unscaled.
Training weights give each compound equal total contribution despite unequal
trajectory counts. C is selected using validation compound-level macro-F1,
then lower NLL, then smaller C. No test scores select hyperparameters, and
there is no encoder fine-tuning or refitting for this public release.

## Aggregation and uncertainty

A classifier predicts one score vector per trajectory. Trajectory probabilities
are averaged equally within each test compound, followed by argmax over the
fixed class order AKT, CDK, HDAC, MAPK, PARP, TUB. Balanced accuracy is the
unweighted mean of the six class recalls. The analysis unit is the compound,
not an image or tile.

All public values are test BA mean ± **sample SD across three seeds**, using
ddof=1. SD is not a confidence interval. Seeds share compounds; plates, images
and seeds are not independent biological replications. Small test cohorts and
selection on previously inspected data limit strong ranking or superiority
claims. Compound/scaffold holdouts do not establish encoder-pretraining
exclusion or held-out-plate/cell-line generalization. The pooled fit uses all
five plates; equal-trajectory averaging gives A549 approximately 60% of the
weight and U2OS 40%, with missing wells changing that mixture.

The fusion row does not establish a conditional image gain beyond both growth
and chemistry. Broader controls and alternative preprocessing are outside this
selected example.

## Provenance and reproduction

Before export, BA was independently recomputed from the original saved
predictions in all 48 contributing fits. Equal-trajectory aggregation, matched
test membership, all three seeds, and aggregate mean/SD were checked. Only the
16 aggregate rows are released; no individual predictions or seed-specific
scores are included. `provenance.json` anchors the source summary digest and
public artifacts without exposing research paths or compound identities.

The plot script reproduces these figures from `metrics.csv`; it does not train
models. The public package provides core preparation, representation, splitting
and classification utilities. Its example CLI does **not** reproduce this
historical matched benchmark end to end: the full locked cohort, saved split
memberships and extraction artifacts are not distributed here.

Dataset: [LCTP](https://doi.org/10.17044/SCILIFELAB.29315642), distributed by its
authors under CC BY-NC-SA 4.0. This release contains no raw microscopy images.
