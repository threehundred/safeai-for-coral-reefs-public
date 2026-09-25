# Safe AI for Coral Reefs

### Benchmarking out-of-distribution detection algorithms for coral reef image surveys

[![Paper](https://img.shields.io/badge/Paper-Ecological%20Informatics%202025-0fa896)](https://doi.org/10.1016/j.ecoinf.2025.103207)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ecoinf.2025.103207-blue)](https://doi.org/10.1016/j.ecoinf.2025.103207)
[![License: CC BY 4.0](https://img.shields.io/badge/Paper%20License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

> 🌐 **[View the project landing page →](https://threehundred.github.io/safe-for-coral-reefs-public/)** (served via GitHub Pages)

**Mathew Wyatt**, Sharyn Hickey, Ben Radford, Manuel Gonzalez-Rivero, Nader Boutros, Nikolaus Callow, Nicole Ryan, Arjun Chennu, Mohammed Bennamoun, James Gilmour

Published in *Ecological Informatics*, Volume 90, December 2025, Article 103207.

---

## Overview

### When AI meets an unfamiliar reef

Deep learning models are increasingly used to automatically classify benthic organisms in coral
reef image surveys. But models trained on one reef, camera, or season can silently fail when they
encounter imagery that differs from their training data — new species, new locations, or shifts in
image quality, colour and blur.

This work treats reliable coral reef classification as a **safety problem**. We benchmark a broad
suite of **out-of-distribution (OOD) detection** algorithms to measure how well they flag imagery
that a model should *not* confidently classify, helping keep automated reef monitoring trustworthy
at scale.

![Dataset Locations](https://ars.els-cdn.com/content/image/1-s2.0-S157495412500216X-gr1.jpg),


## Key contributions

| | |
| --- | --- |
| 📊 **A reef OOD benchmark** | Systematic comparison of ten OOD detection methods on real coral reef survey imagery across multiple regions. |
| 🌊 **Distribution shifts** | Controlled colour and blur perturbations quantify how detectors respond to degraded image quality. |
| 🧭 **Cross-region evaluation** | Test data spans Australian and Pacific reefs — Rowley Shoals, Moorea, and long-term monitoring transects. |
| 📐 **Safety-aware metrics** | Reports AUROC, FPR@95%TPR and calibration (ECE) rather than accuracy alone. |
| 🔁 **Reproducible notebooks** | Jupyter notebooks and pre-computed feature CSVs let you re-run every experiment end to end. |
| 🪸 **Guidance for practitioners** | Practical recommendations for deploying safer AI in operational coral reef monitoring pipelines. |

## Benchmarked out-of-distribution detectors

Detectors are evaluated on features extracted from a trained classifier, using the
[pytorch-ood](https://pytorch-ood.readthedocs.io/) library.

`MaxSoftmax (MSP)` · `ODIN` · `Energy-Based` · `MaxLogit` · `Mahalanobis` · `KNN` · `ViM` · `DICE` · `SHE` · `OpenMax`

## Datasets

The repository ships pre-extracted feature vectors (CSV) for several benthic image datasets, plus
systematically perturbed variants used to simulate distribution shift.

- 🇦🇺 **Pacific / Australian transects** — annotated benthic patches used as in-distribution reference data.
- 🏝️ **Moorea (2008–2010)** — multi-year labelled reef imagery from French Polynesia (a temporal OOD test).
- 🌫️ **Blur shifts** — ten increasing levels of blur applied to NWSS long-term monitoring imagery.
- 🎨 **Colour shifts** — ten increasing levels of colour perturbation simulating changing water/camera conditions.

## Repository contents

| File | Description |
| --- | --- |
| `SafeAI_for_coral_reefs_final.ipynb` | Main analysis & benchmark |
| `pytorchood.ipynb` | OOD detector implementations |
| `openmax.ipynb` | OpenMax experiments |
| `namma_data_ood.ipynb` | Dataset-specific OOD runs |
| `catlin_data_munging.ipynb` | Data preparation & feature prep |
| `data/` | Feature CSVs + blur & colour shifts |

## Detect OOD datasets for your own embeddings

Point the benchmark at your own feature embeddings to measure whether a new survey is
out-of-distribution relative to a reference dataset. High separability (AUROC → 1.0) means the
model should *not* confidently classify that imagery.

```python
import numpy as np
import torch
from pytorch_ood.detector import ODIN, ViM, DICE, KNN
from pytorch_ood.utils import OODMetrics

# 1. Load your own feature embeddings (N x D) + integer labels.
#    Rows are image patches; columns are backbone features
#    (e.g. EfficientNet) — same layout as the CSVs in ./data.
in_X, in_y = load_embeddings("my_reference_reef.csv")   # in-distribution
ood_X, _   = load_embeddings("my_new_survey.csv")        # candidate OOD

# 2. Train a lightweight classifier on the in-distribution data.
model = train_model(in_X, in_y, num_classes=len(np.unique(in_y)))

# 3. Choose which OOD detectors to benchmark.
detectors = {
    "ViM":  ViM(model.features, d=64, w=model.linear.weight, b=model.linear.bias),
    "ODIN": ODIN(model=model, eps=0.001, temperature=1),
    "DICE": DICE(model=model.features, w=model.linear.weight, b=model.linear.bias, p=0.65),
    "KNN":  KNN(model=model.features),
}

# 4. Score in-distribution vs. new data and measure separability.
for name, det in detectors.items():
    det.fit(torch.tensor(in_X))
    scores_in  = det.predict(torch.tensor(in_X))
    scores_out = det.predict(torch.tensor(ood_X))

    metrics = OODMetrics(metrics=["auroc", "fpr95tpr"])
    metrics.update(scores_in,  torch.zeros(len(scores_in)))   # 0 = in-distribution
    metrics.update(scores_out, torch.ones(len(scores_out)))   # 1 = OOD
    result = metrics.compute()
    print(f"{name:5s}  AUROC={result['auroc']:.3f}  FPR95={result['fpr95tpr']:.3f}")

# AUROC ~0.5  -> new data looks in-distribution (safe to classify)
# AUROC ->1.0 -> new data is out-of-distribution (flag for review)
```

## Determine required training data to bring all data in-distribution

Progressively inject a fraction of a new survey into the training pool and retrain, until the
detector can no longer separate it from the reference data. The smallest fraction whose in- and
out-of-distribution score histograms overlap by at least the histogram-intersection threshold is
how much labelled data you need to make that survey safely classifiable.

```python
import numpy as np

# Goal: find how much of a new survey must be added to training
# before that survey is no longer flagged as out-of-distribution.

SAMPLE_SIZE            = 9000
HISTOGRAM_INTERSECTION = 0.9   # "in-distribution" once score histograms overlap

def histogram_intersection(scores_in, scores_out, bins=50):
    # Overlap between the in- and out-of-distribution score distributions.
    # 1.0 = identical (fully in-distribution), 0.0 = perfectly separated (OOD).
    lo = min(scores_in.min(), scores_out.min())
    hi = max(scores_in.max(), scores_out.max())
    h_in,  _ = np.histogram(scores_in,  bins=bins, range=(lo, hi), density=True)
    h_out, _ = np.histogram(scores_out, bins=bins, range=(lo, hi), density=True)
    width = (hi - lo) / bins
    return np.minimum(h_in, h_out).sum() * width

in_X, in_y   = sample_Xy(reference_X, reference_y, n_samples=SAMPLE_SIZE)
new_X, new_y = sample_Xy(survey_X,    survey_y,    n_samples=SAMPLE_SIZE)

required_fraction = None
for fraction in np.arange(0.0, 0.55, 0.05):
    n_inject = int(fraction * SAMPLE_SIZE)

    # Add a slice of the new survey into the training pool.
    if n_inject > 0:
        inj_X, inj_y = sample_Xy(new_X, new_y, n_samples=n_inject)
        train_X = np.concatenate([in_X, inj_X])
        train_y = np.concatenate([in_y, inj_y])
    else:
        train_X, train_y = in_X, in_y

    # Retrain, then score both distributions and measure their overlap.
    model = train_model(train_X, train_y, num_classes=len(np.unique(train_y)))
    scores_in, scores_out = ood_scores(model, in_X=train_X, ood_X=new_X)  # ViM / KNN ...
    hi = histogram_intersection(scores_in, scores_out)

    print(f"added {fraction*100:4.0f}% ({n_inject:5d} samples)  HI={hi:.3f}")
    if hi >= HISTOGRAM_INTERSECTION:
        required_fraction = fraction
        break

if required_fraction is not None:
    print(f"\n{required_fraction*100:.0f}% of the new survey is enough "
          f"to bring it in-distribution.")
else:
    print("\nSurvey stays OOD across the tested range — collect more labels.")
```

## Citation

If you use this code or benchmark in your research, please cite the paper:

```bibtex
@article{wyatt2025safeai,
  title   = {Safe AI for coral reefs: Benchmarking out-of-distribution
             detection algorithms for coral reef image surveys},
  author  = {Wyatt, Mathew and Hickey, Sharyn and Radford, Ben and
             Gonzalez-Rivero, Manuel and Boutros, Nader and Callow, Nikolaus and
             Ryan, Nicole and Chennu, Arjun and Bennamoun, Mohammed and
             Gilmour, James},
  journal = {Ecological Informatics},
  volume  = {90},
  pages   = {103207},
  year    = {2025},
  doi     = {10.1016/j.ecoinf.2025.103207},
  issn    = {1574-9541}
}
```

## License

The repository code is released under the [MIT License](LICENSE). The associated article is
published open access under CC BY 4.0 in *Ecological Informatics*.