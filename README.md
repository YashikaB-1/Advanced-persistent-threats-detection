<div align="center">

# ⚡ APT Detection System

**Hybrid anomaly + supervised detection of Advanced Persistent Threats on network flow data**

Trains an Isolation Forest (unsupervised) and an XGBoost classifier (supervised) on
the CIC-IDS2017 dataset, then fuses their scores into a single risk value and streams
it through a real-time monitoring simulation with latency and drift tracking.

</div>

---

## Table of Contents

- [Overview](#overview)
- [How the Hybrid Model Works](#how-the-hybrid-model-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Real-Time Detection](#real-time-detection)
- [Web Dashboards](#web-dashboards)
- [Design Notes](#design-notes)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

APTs hide inside normal-looking traffic, so a single model rarely covers every case:

- A **supervised** model (XGBoost) is accurate on attack patterns it has seen.
- An **unsupervised** model (Isolation Forest) flags anything that deviates from
  normal, including patterns it was never trained on.

This project trains both on [CIC-IDS2017](#dataset) network flows and combines them:

```
risk_score = 0.6 · XGBoost P(attack)  +  0.4 · normalized IsolationForest anomaly score
```

A record is escalated as **HIGH** (`> 0.8`), **MEDIUM** (`> 0.6`), or **NORMAL** otherwise.

---

## How the Hybrid Model Works

`src/train.py`:

1. **Load** `data/processed.csv` (already numeric + scaled).
2. **Drop leakage / overpowered features** — `Flow Bytes/s`, `Flow Packets/s`,
   `Packet Length Variance`, `Avg Packet Size`, `Fwd/Bwd Packet Length Std`,
   `Init_Win_bytes_forward`. These let a model "cheat" and inflate scores.
3. **Time-based 80/20 split** — first 80% of rows train, last 20% test. No shuffle,
   so the model is never evaluated on data that precedes its training window.
4. **Add Gaussian noise** (σ = 0.01) to the training features to curb overfitting.
5. **Isolation Forest** — `contamination=0.05`, fit **only on normal rows**
   (`y_train == 0`) so it learns a model of benign traffic.
6. **XGBoost** — regularised: `max_depth=3`, `n_estimators=50`, `subsample=0.7`,
   `colsample_bytree=0.7`, `reg_lambda=2`, `reg_alpha=1`.
7. **Fuse** — normalise the Isolation Forest scores to `[0, 1]`, then
   `final = 0.6·xgb + 0.4·iso`; threshold at `0.6` for the binary decision.
8. **Report** — classification report, confusion matrix, and business metrics
   (detection rate, false-positive rate) for the hybrid model, plus a comparison
   against standalone Logistic Regression, Random Forest, and XGBoost.
9. **Save** `models/iso.pkl` and `models/xgb.pkl`.

---

## Project Structure

```
apt-detection/
├── src/
│   ├── combine.py            Concatenate the 8 raw CIC-IDS2017 CSVs (50k rows each)
│   ├── balance.py            Binary labels (BENIGN=0, else=1) + class balancing
│   ├── preprocess.py         Drop ID cols, coerce numeric, drop NaN/inf, StandardScaler
│   ├── train.py              Hybrid model (Isolation Forest + XGBoost) + comparison
│   ├── realtime.py           Console streaming simulation over processed.csv
│   ├── realtime_advanced.py  + latency tracking, feature-drift detection, alert summary
│   └── test_columns.py       Debug helper — prints dataset columns
│
├── app.py                    Flask app (loads models + data at startup)
├── streamlit_app.py          Streamlit dashboard — live table, metrics, latency chart
├── templates/
│   └── index.html            Flask front-end (terminal theme)
│
├── models/                   Trained artifacts — iso.pkl, xgb.pkl
├── data/                     Dataset — NOT committed, see data/README.md
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<you>/apt-detection.git
cd apt-detection

python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Dataset

**CIC-IDS2017** (Canadian Institute for Cybersecurity). It is not stored in the
repo — see [`data/README.md`](data/README.md) for the download link and the exact
folder layout. Drop the eight `*_ISCX.csv` files into `data/MachineLearningCVE/`,
then run the [pipeline](#pipeline) to regenerate the derived CSVs.

---

## Pipeline

> All `src/` scripts use paths **relative to `src/`**, so run them from inside that folder.

```bash
cd src
python combine.py       # data/MachineLearningCVE/*.csv  → data/dataset_small.csv
python balance.py       # data/dataset_small.csv         → data/dataset_balanced.csv
python preprocess.py    # data/dataset_balanced.csv      → data/processed.csv
python train.py         # data/processed.csv             → models/iso.pkl, models/xgb.pkl
```

`train.py` prints the hybrid model's classification report, confusion matrix,
detection rate / false-positive rate, and a side-by-side comparison with the
baseline models.

---

## Real-Time Detection

Simulates a live flow stream by iterating over `processed.csv`, scoring each record
with the hybrid model, and printing an alert level.

```bash
cd src
python realtime.py            # basic: score → NORMAL / MEDIUM / HIGH, 0.3s tick
python realtime_advanced.py   # + per-record latency (ms), feature-drift flag, final summary
```

`realtime_advanced.py` compares each record's feature means against the training
mean and raises `⚠️ DRIFT` when the average absolute difference exceeds `0.5`.
Both scripts append JSON lines to `logs.txt` (git-ignored).

---

## Web Dashboards

### Streamlit (recommended)

```bash
streamlit run streamlit_app.py     # run from the repo root
```

Pick a traffic mix (Mixed / Normal Only / Attack Heavy), start monitoring, and
watch a live table, HIGH/MEDIUM/NORMAL counters, a running line chart, and a
latency readout.

### Flask

```bash
python app.py                      # run from the repo root → http://127.0.0.1:5000
```

Loads `models/*.pkl` and `data/processed.csv` at startup and exposes `/` (the
dashboard) and `/start` (returns the first 100 scored records as JSON).

> **Known issue:** `templates/index.html` calls a `/stream` Server-Sent-Events
> endpoint that `app.py` does not implement (it exposes `/start` instead), so the
> in-page button is currently inert. The Streamlit app is the working UI; wiring
> `app.py` to SSE — or pointing the template at `/start` — is a good first fix.

---

## Design Notes

- **Why time-based split?** Shuffling network-flow data leaks future patterns into
  training and produces unrealistically high scores. Splitting chronologically is
  the honest evaluation.
- **Why drop those 7 features?** Rate and packet-size-variance features correlate
  almost perfectly with the label in CIC-IDS2017; keeping them makes every model
  look near-perfect and hides real behaviour.
- **Why a hybrid?** The Isolation Forest contributes recall on anomalous-but-unseen
  traffic; XGBoost contributes precision on known attack signatures. The 0.6 / 0.4
  weighting favours the supervised signal while keeping the anomaly term meaningful.

---

## Limitations

- The "stream" is a replay of a static CSV, not live packet capture.
- `preprocess.py` fits `StandardScaler` on the full balanced set before the
  train/test split in `train.py`, so scaling statistics see the test rows.
- Class balancing downsamples the majority class, discarding a large share of
  benign traffic.
- Only a binary decision (attack / normal) — attack **type** is collapsed away in
  `balance.py`.

---

## License

CIC-IDS2017 is distributed by the Canadian Institute for Cybersecurity, University of New Brunswick.
