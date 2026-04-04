# 🏙️ Real Estate Intelligence Platform — Faridabad

An end-to-end real estate analytics platform that scrapes, scores, and analyzes property listings from the Faridabad market. Built to demonstrate applied ML, NLP, and full-stack data engineering in a real-world domain.

---

## 📊 Evaluation Results

| Metric | Value |
|---|---|
| Builder Floors — RMSE | ₹751 / sqft |
| Builder Floors — R² | 0.74 |
| Plots — R² | 0.84 |
| Societies — R² | 0.64 |
| RAG Faithfulness | 1.00 / 1.0 |
| RAG Answer Relevancy | 0.79 / 1.0 |

---

## 🏗️ Architecture

```
housing.com
    │
    ▼
scraper.py          ← SeleniumBase scraper (plots, floors, societies)
    │
    ▼
PostgreSQL DB       ← Structured property data via SQLAlchemy
    │
    ├──► scoring.py     ← Alpha scoring engine (haversine + infra drivers)
    │
    ├──► train.py       ← XGBoost model training (per category)
    │
    ├──► eval.py        ← RMSE/R² + RAG faithfulness/relevancy evaluation
    │
    ├──► ingest.py      ← PDF ingestion → ChromaDB (OCR + embeddings)
    │
    └──► app.py         ← Streamlit dashboard (heatmap, deal scorer, RAG chat)
```

---

## ✨ Features

### 🔍 Web Scraper (`scraper.py`)
- Scrapes plots, builder floors and societies from housing.com using SeleniumBase
- Handles multi-BHK listings (e.g. "2, 3 BHK Flats") — saves separate rows per BHK type
- Deep scans individual listing pages for area, possession status, amenities
- Deduplication, price cleaning and sector extraction built-in

### 🧠 Alpha Scoring Engine (`scoring.py`)
- Calculates `connectivity_score` for every property using haversine distance
- 15 infrastructure drivers — metro stations, expressway exits, Jewar Airport link
- Infrastructure maturity weights (existing = 1.0x, upcoming = 0.6x)
- Expressway proximity bonus (+20% for properties within 2km)
- Category-specific alpha ratings: Plots (HIGH>7.0), Floors (HIGH>7.5), Societies (HIGH>8.0)

### 📈 XGBoost Deal Scorer (`train.py` + `app.py`)
- Separate XGBoost regression model per property category
- Features: sector, area, BHK type, connectivity score
- Conservative hyperparameters (max_depth=3, min_child_weight=5) for small dataset
- Real-time deal scoring in UI — flags listings as 🟢 Good Deal / 🟡 Fair / 🔴 Overpriced
- Shows % deviation from both ML-predicted fair rate and sector average

### 🤖 RAG Pipeline (`ingest.py` + `app.py`)
- Ingests property brochures, RERA certificates and legal documents (PDF)
- OCR fallback via pytesseract for image-based/scanned PDFs
- ChromaDB vector store with OpenAI `text-embedding-3-small` embeddings
- Separate collections per document type (brochure / rera / legal)
- GPT-4o powered chat with source chunk transparency
- Grounded in Faridabad real estate context via system prompt

### 📊 Streamlit Dashboard (`app.py`)
- 3D investment heatmap (Pydeck ColumnLayer) — column height = alpha score
- Dynamic filters: sector, price range, BHK type, alpha rating
- Market phase labels: 🌟 ELITE / 🔥 PRIME / 📈 GROWTH / 🌱 EMERGING
- Top Alpha Opportunities panel with connectivity score progress bars
- CSV export for filtered inventory

---

## 🗂️ Project Structure

```
├── scraper.py              # Web scraping engine
├── scoring.py              # Alpha scoring engine  
├── train.py                # XGBoost model training
├── eval.py                 # ML + RAG evaluation
├── ingest.py               # PDF ingestion + RAG pipeline
├── app.py                  # Streamlit dashboard
├── models.py               # SQLAlchemy ORM models
├── database.py             # DB connection
├── populate_infra.py       # Infrastructure drivers seeder
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/real-estate-intelligence-platform.git
cd real-estate-intelligence-platform
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your keys
```

### 5. Set up PostgreSQL database
```bash
# Create a database named 'real_estate' in PostgreSQL
# Then run:
python -c "from database import Base, engine; from models import *; Base.metadata.create_all(engine)"
```

### 6. Seed infrastructure drivers
```bash
python populate_infra.py
```

### 7. Scrape property data
```bash
python scraper.py
# Choose 1 (Plots), 2 (Builder Floors), or 3 (Societies)
```

### 8. Run alpha scoring
```bash
python scoring.py
```

### 9. Train ML models
```bash
python train.py
```

### 10. Launch the app
```bash
streamlit run app.py
```

### 11. (Optional) Run evaluation
```bash
python eval.py              # Full evaluation
python eval.py --skip-rag   # ML only
```

---

## 🔧 Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (for RAG + eval) |
| `DATABASE_URL` | PostgreSQL connection string |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `seleniumbase` | Web scraping |
| `sqlalchemy` | ORM + PostgreSQL |
| `xgboost` | Property valuation model |
| `scikit-learn` | Model evaluation |
| `chromadb` | Vector store for RAG |
| `openai` | Embeddings + GPT-4o |
| `pdfplumber` | PDF text extraction |
| `pytesseract` | OCR for scanned PDFs |
| `streamlit` | Dashboard UI |
| `pydeck` | 3D investment heatmap |
| `pandas` / `numpy` | Data processing |

---

## 📝 Notes

- Dataset is intentionally kept at ~200 rows per category (static snapshot, not live)
- ML model accuracy improves with more data — societies model benefits most from additional scraping
- ChromaDB uses persistent storage — ingested documents survive app restarts
- Tesseract OCR must be installed separately on Windows: [Download here](https://github.com/UB-Mannheim/tesseract/wiki)

---

## 👤 Author

**Prakhar Sharma**  
[LinkedIn](https://linkedin.com/in/prakhar-sharma-b4a53a246/) · [GitHub](https://github.com/Prakharr05)