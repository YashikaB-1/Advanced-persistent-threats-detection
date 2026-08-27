# Data

The dataset is **not committed** (raw files total ~1.1 GB, and several exceed
GitHub's 100 MB per-file limit). Download it and regenerate the derived CSVs
with the pipeline.

## 1. Download CIC-IDS2017

Source: Canadian Institute for Cybersecurity —
<https://www.unb.ca/cic/datasets/ids-2017.html>
(also mirrored on Kaggle as "CICIDS2017").

Grab the **machine-learning CSV** distribution (the `MachineLearningCVE` folder)
— eight labelled flow files:

```
data/MachineLearningCVE/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## 2. Regenerate the derived files

From the `src/` folder, in order:

```bash
cd src
python combine.py      # → data/dataset_small.csv     (50k rows/file, concatenated)
python balance.py      # → data/dataset_balanced.csv  (binary labels, class-balanced)
python preprocess.py   # → data/processed.csv         (numeric, StandardScaler)
```

`processed.csv` is what `train.py` and the apps consume.
