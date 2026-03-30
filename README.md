# Amazon Product Analysis & Recommendation System

## Project Structure

```
├── amazon.csv                # Dataset (place here)
├── amazon_analysis.ipynb     # EDA Jupyter Notebook
├── app.py                    # Streamlit Recommendation App
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 1. EDA Notebook — `amazon_analysis.ipynb`

Open with Jupyter:
```bash
jupyter notebook amazon_analysis.ipynb
```

### Sections Covered
| # | Section | What it shows |
|---|---------|--------------|
| 1 | Setup & Loading | Load CSV, preview schema |
| 2 | Cleaning & Feature Engineering | Clean prices, create value_score, segments |
| 3 | Descriptive Stats | mean/std/quartiles for numeric columns |
| 4 | Category Distribution | Bar + Pie by main/sub category |
| 5 | Price Analysis | Histograms, category price comparison, scatter |
| 6 | Rating Analysis | Distribution, KDE, boxplot, correlation heatmap |
| 7 | Discount Deep Dive | Violin plot, discount brackets |
| 8 | Top Products | Top by value score, savings, reviews |
| 9 | Review Word Frequency | Top 30 words in review titles |
| 10 | Price Segments | Budget → Luxury breakdown with avg ratings |
| 11 | Summary | Printed stats summary |

---

## 2. Recommendation App — `app.py`

### Installation
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

> Make sure `amazon.csv` is in the **same folder** as `app.py`.

### Recommendation Modes

| Mode | How it works |
|------|-------------|
| **Search & Similar** | Search by keyword → pick a product → get TF-IDF cosine-similarity recommendations |
| **Category Explorer** | Browse all products in a category, sorted by your chosen metric |
| **Budget Finder** | Set a price range (and optional category) → get best-value products |
| **Top Rated** | Leaderboard ranked by value score, rating, reviews, or savings |

### Recommendation Algorithm
- **Feature source**: product name (×3 weight) + main category (×2) + product description
- **Vectorisation**: TF-IDF (pure NumPy, no sklearn required)
- **Similarity**: Cosine similarity via dot product of L2-normalised vectors
- **Value score**: `rating × log(1 + review_count)` — balances quality and popularity

---

## Dependencies
- Python 3.8+
- pandas, numpy
- streamlit
- matplotlib, seaborn (notebook only)
