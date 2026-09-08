# Building Age Classification from Satellite Imagery

Hack4Dev hackathon challenge: classify the majority construction-age class (1–4) of
buildings in a 30 m Landsat pixel from its spectral reflectance time series, then
transfer a Madrid-trained model to Amsterdam with few labels.

## Project layout

```
.
├── notebooks/                     Run in order; each assumes the repo root as ../
│   ├── 1-Introduction.ipynb       Challenge, dataset, and age-class definitions
│   ├── 2-Reading_Data.ipynb       Load parquet, explore features/labels/time series
│   ├── 3-Preprocessing.ipynb      Feature engineering → data/preprocessed/
│   └── 4-Modelling.ipynb          Metric-learning model, CV, zero-/few-shot transfer
├── src/
│   └── data.py                    Reusable pipeline: lean loading → 60 features
├── data/                          Input data (git-ignored, not committed)
│   ├── madrid_train.parquet       Stage 1 training city
│   ├── amsterdam_data.parquet     Transfer target city
│   └── preprocessed/              Written by notebook 3, read by notebook 4
├── docs/
│   ├── PROJECT_GUIDE.md              Plain-language walkthrough of the whole project
│   ├── MASTERCLASS.md                Deep dive + how the work maps to the grading rubric
│   ├── RELATED_WORK.md               Published prior art, benchmark results, methods to steal
│   ├── METRIC_LEARNING_APPROACH.md   Strategy write-up
│   └── Evaluation Rubric Overview.docx   Organisers' grading rubric
└── requirements.txt
```

New to the project? Start with [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) — it explains
every notebook, the data, and the modelling approach in simple terms. Then read
[docs/MASTERCLASS.md](docs/MASTERCLASS.md) for the technical deep dive and how each stage
scores against the rubric.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `madrid_train.parquet` and `amsterdam_data.parquet` in `data/`, then run the
notebooks from the `notebooks/` folder in numerical order.
