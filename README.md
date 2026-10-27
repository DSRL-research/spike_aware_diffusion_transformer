# Spike-aware diffusion transformer for energy forecasting

Code for the paper *Spike-aware diffusion transformer model for energy forecasting* (A. Mateescu, T. Cioara, G. Papias,
V. Michalakopoulos, E. Sarmas, I. Anghel).

## Setup

Python 3.12 and a CUDA GPU (the experiments ran on an NVIDIA GeForce RTX 4060 Ti, 8 GB).

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
bash scripts/fetch_third_party.sh      # official code of the baselines, only needed for DLinear, TimeGrad and CSDI
```

`fetch_third_party.sh` downloads the authors' code, unchanged, into `third_party/`: CSDI (ermongroup/CSDI, commit
`7f24a43`), TimeGrad (zalandoresearch/pytorch-ts, release v0.6.0, with two modules of gluonts 0.9.0) and DLinear
(cure-lab/LTSF-Linear, commit `0c11366`). The adapters in `src/baselines/` feed them the same windows, features and
scaling as our models.

## Data

Put the raw files in `data/raw/` and build the processed tables:
**Heat demand.** `data/processed/heat.parquet` is included. It is derived from the district heat demand data of
De Mulder (2023), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: outages removed
from the target and filled causally in the history; calendar and target-history features added (`data/prepare_heat.py`).

The weather variables used by the heat spike detector (temperature, humidity, wind speed, cloud cover, rainfall, solar
irradiation) come from a third-party source that cannot be redistributed, so they are not included. The heat
forecasters and baselines do not use weather and run on the included table. Retraining the heat detector
(`train_detector heat`, `sensitivity_k`) requires adding hourly weather for the same period, as the columns
`Temperature`, `Humidity`, `Wind Speed`, `Cloud Cover`, `Rainfall` and `Solar Irradiation` of
`data/raw/heatDemand.csv` (with `Time` and `Power`), then running `python -m data.prepare_heat`. With weather from
another source, the detector results will differ from the paper.

**Wind power.** Put `Vestas_V52_preprocessed_history_data.csv` (10-minute `Timestamp`, `WindSpeed`,
`StdDevWindSpeed`, `WindDirection`, `ActivePower` (MW), `TheoreticalPower` (MW), `Pitch`, `GenRPM`, `EnvirTemp`) in
`data/raw/` and run `python -m data.prepare_wind`. Source: Vestas V52 turbine SCADA data (Byrne and MacArtain, 2022)
with the theoretical power of Michalakopoulos et al.

| dataset | raw file | source |
|---|---|---|
| heat demand | `heatDemand.csv`: hourly `Time` (UTC), `Power`, `Temperature`, `Humidity`, `Wind Speed`, `Cloud Cover`, `Rainfall`, `Solar Irradiation` | De Mulder (2023), district heat demand of a city district in Belgium |
| wind power | `Vestas_V52_preprocessed_history_data.csv`: 10-minute `Timestamp`, `WindSpeed`, `StdDevWindSpeed`, `WindDirection`, `ActivePower` (MW), `TheoreticalPower` (MW), `Pitch`, `GenRPM`, `EnvirTemp` | Vestas V52 turbine SCADA data (Byrne and MacArtain, 2022) with the theoretical power of Michalakopoulos et al. |

```bash
python -m data.prepare_heat
python -m data.prepare_wind
```

`checkpoints/heat/` holds the trained heat-demand models of the paper: the spike detector (`detector.pt`, seed 0), its
probabilities for every window (`probabilities.npz`) and, for seeds 42-46, the diffusion transformer with its noise
sampler (`model.pt`, `noise.pt`) for diffusion only, S2 and S3. S1 uses the diffusion-only model and the detector
probabilities, so it has no checkpoint of its own. To reproduce the heat results of the paper without training:

```bash
python -m scripts.evaluate_pretrained
```

This writes the validation and test predictions of diffusion only, S1, S2 and S3 to `results/heat/`, in the same
layout as steps 1 and 2, and prints the test table. `python -m scripts.ablation_gmm` then runs without training. With
the weather columns added (see Data), `--detector` recomputes the probabilities and the detector metrics from
`detector.pt`.
## Reproducing the results

All commands run from the repository root and write to `results/<dataset>/`. Run them in this order for each dataset

| step | command | results |
|---|---|---|
| 1 | `python -m scripts.train_detector <dataset>` | spike detector, its probabilities for every window, classification metrics (AUROC, AUPRC, precision, recall and F1 at the validation-optimal threshold, Brier skill, ECE) |
| 2 | `python -m scripts.train_forecasters <dataset>` | transformer baseline, diffusion only, S1, S2, S3: test predictions and the overall and spike-point metrics |
| 3 | `python -m scripts.run_dlinear <dataset>`, `python -m scripts.run_timegrad <dataset>`, `python -m scripts.run_csdi <dataset>` | state-of-the-art baselines |
| 4 | `python -m scripts.significance <dataset>` | paired Wilcoxon signed-rank tests of S1-S3 against every baseline (Holm-corrected, rank-biserial effect sizes) |

Ablations (after steps 1 and 2):

| command | results |
|---|---|
| `python -m scripts.ablation_baseline_s1` | S1 applied to the transformer baseline and to diffusion only |
| `python -m scripts.fit_noise_families`, then `python -m scripts.ablation_noise_family` | Gaussian and Student-t noise instead of Johnson-SU, with Wilcoxon tests against Johnson-SU |
| `python -m scripts.ablation_gmm` | detector probabilities replaced by draws of a Gaussian mixture fitted to them (no timing information) |
| `python -m scripts.sensitivity_k` | spike threshold k of the detector labels (1.0, 1.5, 2.0, 2.5) |

## Settings

| | heat demand | wind power |
|---|---|---|
| history / horizon | 168 h / 24 h | 72 h / 24 h |
| split (chronological, 24-window gaps) | 70 / 20 / 10 % | 60 / 20 / 20 % |
| spike | target > history mean + 1.5 history std | same |
| S1 (kappa, beta) | (0.1, 0.75) | (0.9, 0.1) |
| Eq. 32 scale for S2 / S3 | 0.2 / 0.2 | 0 / 0 |
| seeds | forecasters and baselines 42-46, detector 0; noise-family and k studies 42-44 | forecasters and baselines 42-46, detector 0 |

All hyperparameters are in `configs/heat.py`, `configs/wind.py` and `configs/base.py`. Metrics are computed on the test
split after averaging the overlapping forecasts of each hour; wind forecasts are clipped at zero. Training uses
deterministic GPU algorithms and fixed seeds, so reruns on the same hardware and software give the same numbers, except
TimeGrad, whose dilated convolutions run with non-deterministic cuDNN kernels (the deterministic ones are about 200 times
slower).

## Layout

```
configs/      dataset settings and model hyperparameters
data/         preparation of the processed tables
src/          datasets, models, training, sampling, metrics, integration strategies, baseline adapters
scripts/      experiments
```
## License

The code is released under the MIT License (`LICENSE`). The heat-demand data in `data/processed/` is derived from
De Mulder (2023) and remains under CC BY 4.0.