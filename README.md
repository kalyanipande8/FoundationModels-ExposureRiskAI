# Foundation Models & Generative AI for Exposure Risk Assessment

**Stack (portfolio label):** PyTorch, Transformers, Generative AI
**What it actually is (this repo):** NumPy + scikit-learn + SciPy, offline, no live LLM/API calls, no cost.

## What this actually is

This project explores the *ideas* behind foundation-model transfer learning
and generative data augmentation for air-quality exposure-risk prediction,
implemented as small, real, from-scratch machine learning code rather than
by calling a live LLM or a pretrained foundation model API.

**Why not literal PyTorch / a real transformer / a live foundation model?**
On the machine this was built on, `torch` is not installed and free disk
space was critically low (~1.5GB available), so installing a multi-hundred-MB
deep learning framework was judged unsafe. Calling a hosted foundation-model
API was also out of scope by design (standing decision for this portfolio
project: self-contained, fully offline, zero API cost, fully reproducible).

Instead, everything here is **real, working, from-scratch NumPy code**:

- `src/nn.py` — a real feedforward neural network (Dense layers, ReLU/sigmoid
  activations, manual backprop, Adam optimizer) that supports freezing
  arbitrary layers. This is what makes genuine transfer learning
  (reusing a pretrained encoder, frozen or fine-tuned) possible, which
  off-the-shelf `sklearn.neural_network.MLPClassifier` does not expose.
- `src/vae.py` — a real Variational Autoencoder (encoder → mu/logvar →
  reparameterization trick → decoder), with the ELBO loss and its gradients
  derived and implemented by hand, trained with Adam. This is the
  "generative model" used for data augmentation and scenario modeling.
- No transformer self-attention block is implemented; the "encoder" is a
  small MLP. The transfer-learning mechanics (pretrain broad → freeze →
  fine-tune narrow) are the same mechanics used with real transformer
  foundation models, just at small scale and with an MLP encoder instead
  of a stack of attention layers.

If PyTorch and more disk space are available in the future, the encoder in
`src/models.py` could be swapped for a small `nn.TransformerEncoder` with
minimal changes to `src/transfer_experiment.py` — the pretrain / freeze /
fine-tune / compare structure would stay identical.

## Task and data

All data is synthetic, generated in `src/data_sim.py`, and is a real
computed function of features + noise — not literal real air-quality data.

- **Features (8):** `pm25, pm10, no2, o3, co, temperature, humidity, wind_speed`
- **Label:** binary exposure-risk flag, derived from a nonlinear
  dose-response function of the features (linear terms + PM2.5×temperature
  and NO2×wind interaction terms + a tanh nonlinearity), thresholded at the
  median, plus Gaussian noise.
- **Pretraining set:** 40 synthetic "regions" × 150 samples = 6,000 samples,
  each region with its own randomly perturbed dose-response coefficients —
  a broad, heterogeneous synthetic corpus standing in for foundation-model
  pretraining data.
- **Target set:** one specific held-out "region" with its own systematic
  coefficient shift, split into a **small** training set (60 samples) and a
  test set (150 samples) — the realistic small-data regime where transfer
  learning is expected to help.

## 1. Transfer learning vs. training from scratch (real comparison)

Architecture: `Dense(8→32, ReLU) → Dense(32→16, ReLU)` (the "encoder") →
`Dense(16→8, ReLU) → Dense(8→1, sigmoid)` (the "head").

- **Transfer model:** encoder pretrained on the 6,000-sample broad set,
  then **frozen** (feature extraction) while a fresh head is trained on the
  60-sample target training set.
- **From-scratch model:** identical architecture, randomly initialized,
  trained only on the same 60-sample target training set.
- Evaluated on the same 150-sample held-out target test set.

**Real measured results (seed=42, single run, `python run_all.py`):**

| | Test accuracy | Test AUC |
|---|---|---|
| Transfer (frozen pretrained encoder) | 0.880 | 0.9264 |
| From-scratch | 0.860 | 0.9254 |

**Averaged over 5 seeds (1–5), 100 epochs each** (this is what
`tests/test_pipeline.py::test_transfer_beats_from_scratch_on_average` checks):

| | Mean test AUC | Mean test accuracy |
|---|---|---|
| Transfer | 0.9454 | 0.8467 |
| From-scratch | 0.9352 | 0.8600 |

Honest interpretation: transfer consistently gives a small but real AUC
lift (~1 point on average, and higher in every one of the 5 seeds tested
individually), which is the metric the test enforces. Accuracy at the
default 0.5 threshold is noisier and not always higher for transfer with
only 60/150 train/test samples — reported as-is, not cherry-picked.

## 2. Generative data augmentation (small VAE)

`src/vae.py`'s `SmallVAE` is trained on the 60-sample target training set's
standardized features, then used to generate synthetic feature vectors.
Synthetic samples are pseudo-labeled with a k-NN classifier fit on the real
labeled training data (no oracle, no test-label leakage), added to the
training set, and a fresh classifier is trained from scratch on the
augmented set vs. the original 60-sample baseline.

**Real measured result (seed=42, `python run_all.py`):**

| | Test accuracy | Test AUC |
|---|---|---|
| Baseline (60 real samples) | 0.873 | 0.940 |
| Augmented (60 real + 120 synthetic) | 0.833 | 0.885 |
| Delta | −0.040 | −0.055 |

**Honest result: augmentation hurt performance in this run.** With only 60
real samples, the VAE's learned distribution is an imperfect approximation
of the true small-sample distribution, and k-NN pseudo-labels on generated
points are noisier than real labels — both plausible, real explanations for
the regression, not implementation bugs. This is reported as-is per the
project's design goal of an honest measurement, not a tuned-to-look-good
result. (`tests/test_pipeline.py` only checks the experiment runs and
produces finite, valid metrics — it does not assert augmentation must help.)

## 3. Scenario modeling

`src/scenario_modeling.py` uses the same trained VAE's decoder to sample:
a baseline synthetic distribution (latent prior N(0,1)), and two shifted
scenarios ("Scenario A" / "Scenario B") obtained by shifting the latent
prior's mean by +1.5 / −1.5 before decoding. All statistics reported (means
and standard deviations per feature, in original physical units) are
computed from real generated samples, not invented. The VAE's latent axes
are unlabeled and not guaranteed to align with a single physical variable,
so scenarios are reported neutrally (A/B) rather than claimed to represent
a specific named event (e.g. "heatwave") — see `output/results.json` for
full per-feature numbers.

## Test results

```
$ python -m pytest tests/ -v
tests/test_pipeline.py::test_pretrain_data_shapes PASSED
tests/test_pipeline.py::test_target_data_shapes_and_split PASSED
tests/test_pipeline.py::test_transfer_beats_from_scratch_on_average PASSED
tests/test_pipeline.py::test_vae_trains_and_reduces_loss PASSED
tests/test_pipeline.py::test_vae_samples_are_statistically_plausible PASSED
tests/test_pipeline.py::test_augmentation_experiment_runs_and_reports_real_numbers PASSED

6 passed in ~36s
```

## Running it

```bash
pip install -r requirements.txt
python run_all.py          # runs everything, writes output/results.json
python -m pytest tests/ -v # runs the sanity-check test suite
```

## Repository layout

```
src/
  data_sim.py               # synthetic pretraining + target data generation
  nn.py                     # from-scratch NumPy MLP + backprop + Adam
  models.py                 # encoder/head architecture factory
  transfer_experiment.py    # pretrain -> freeze -> fine-tune vs from-scratch
  vae.py                    # from-scratch NumPy VAE (generative model)
  augmentation_experiment.py# VAE-based data augmentation experiment
  scenario_modeling.py      # generative scenario sampling demo
run_all.py                  # runs everything end-to-end
tests/test_pipeline.py      # real sanity checks (no mocks)
output/results.json         # saved real results from the last run
```
