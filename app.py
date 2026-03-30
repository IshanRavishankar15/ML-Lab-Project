import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon=":shopping_cart:",
    layout="wide",
)

st.markdown("""
<style>
    .product-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-val { font-size: 24px; font-weight: 700; }
    .metric-lbl { font-size: 12px; opacity: 0.85; margin-top: 2px; }
    .price-disc { font-size: 20px; font-weight: 700; color: #B12704; }
    .price-orig { font-size: 13px; color: #888; text-decoration: line-through; }
    .disc-tag   { color: #007600; font-weight: 600; font-size: 13px; }
    .stars      { color: #f4a340; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """
    Loads pre-cleaned amazon_clean.csv (produced by the EDA notebook).
    Falls back to raw amazon.csv with on-the-fly cleaning if the clean file is absent.
    """
    if os.path.exists("amazon_clean.csv"):
        df = pd.read_csv("amazon_clean.csv")
    else:
        st.warning(
            "`amazon_clean.csv` not found — falling back to raw `amazon.csv`. "
            "Run the EDA notebook first to generate the clean file."
        )
        df = pd.read_csv("amazon.csv")

        def clean_price(col):
            return pd.to_numeric(
                col.astype(str).str.replace("[₹,]", "", regex=True).str.strip(),
                errors="coerce",
            )

        df["discounted_price"]   = clean_price(df["discounted_price"])
        df["actual_price"]       = clean_price(df["actual_price"])
        df["rating"]             = pd.to_numeric(df["rating"], errors="coerce")
        df["discount_pct"]       = pd.to_numeric(
            df["discount_percentage"].astype(str).str.replace("%", "").str.strip(),
            errors="coerce",
        )
        df["rating_count_clean"] = pd.to_numeric(
            df["rating_count"].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        )
        df["main_category"] = df["category"].str.split("|").str[0]
        df["sub_category"]  = df["category"].str.split("|").str[1]
        df["savings"]       = df["actual_price"] - df["discounted_price"]
        df["value_score"]   = df["rating"] * np.log1p(df["rating_count_clean"].fillna(0))

    df = df.dropna(subset=["actual_price", "discounted_price", "rating"])
    return df.reset_index(drop=True)

# TF-IDF RECOMMENDATION ENGINE 

@st.cache_data(show_spinner="Building recommendation engine…")
def build_tfidf_matrix(df):
    """
    TF-IDF vectors from product name (x3), category (x2), and description.
    Rows are L2-normalised so dot-product == cosine similarity.
    """
    def tokenize(text):
        return re.findall(r"[a-z0-9]+", str(text).lower())

    corpus = [
        tokenize(r["product_name"]) * 3
        + tokenize(r["main_category"]) * 2
        + tokenize(str(r.get("about_product", "")))
        for _, r in df.iterrows()
    ]

    all_terms = sorted({t for doc in corpus for t in doc})
    term_idx  = {t: i for i, t in enumerate(all_terms)}
    n_docs, n_terms = len(corpus), len(all_terms)

    tf = np.zeros((n_docs, n_terms), dtype=np.float32)
    for d, tokens in enumerate(corpus):
        cnt   = Counter(tokens)
        total = sum(cnt.values())
        for t, c in cnt.items():
            if t in term_idx:
                tf[d, term_idx[t]] = c / total

    df_count = (tf > 0).sum(axis=0).astype(np.float32)
    idf      = np.log((n_docs + 1) / (df_count + 1)) + 1.0
    tfidf    = tf * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tfidf / norms


def content_based_recommend(df, tfidf, product_idx, top_n=10):
    sims = tfidf @ tfidf[product_idx]
    sims[product_idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_n]
    result  = df.iloc[top_idx].copy()
    result["similarity"] = sims[top_idx]
    return result


def category_top_products(df, category, sort_by="value_score", top_n=10):
    return df[df["main_category"] == category].nlargest(top_n, sort_by)


def price_range_recommend(df, min_price, max_price, category=None, top_n=10):
    mask = (df["discounted_price"] >= min_price) & (df["discounted_price"] <= max_price)
    sub  = df[mask]
    if category and category != "All":
        sub = sub[sub["main_category"] == category]
    return sub.nlargest(top_n, "value_score")

# UI components

def star_html(rating):
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def render_product_card(row, idx=0, show_sim=False, sim_val=None):
    name     = str(row.get("product_name", "N/A"))
    cat      = row.get("main_category", "")
    sub_cat  = row.get("sub_category", "")
    rating   = float(row.get("rating", 0))
    rc       = row.get("rating_count_clean", 0)
    disc     = float(row.get("discounted_price", 0))
    orig     = float(row.get("actual_price", 0))
    disc_pct = float(row.get("discount_pct", 0))
    link     = row.get("product_link", "#")

    rc_str  = f"{int(rc):,}" if pd.notna(rc) else "N/A"
    sim_str = (
        f' &nbsp;<span style="font-size:12px;color:#555;">sim {sim_val:.2f}</span>'
        if show_sim and sim_val is not None else ""
    )
    sub_str = (
        f'<span style="font-size:12px;color:#555;"> · {str(sub_cat)[:30]}</span>'
        if sub_cat and pd.notna(sub_cat) else ""
    )

    st.markdown(f"""
    <div class="product-card">
      <div style="font-size:15px;font-weight:700;margin-bottom:5px;color:black">
        {idx}. {name[:95]}{'…' if len(name) > 95 else ''}
      </div>
      <div style="font-size:12px;color:#444;margin-bottom:7px;">
        {cat}{sub_str}{sim_str}
      </div>
      <div>
        <span class="price-disc">₹{disc:,.0f}</span>
        &nbsp;<span class="price-orig">₹{orig:,.0f}</span>
        &nbsp;<span class="disc-tag">{disc_pct:.0f}% off</span>
      </div>
      <div style="margin-top:5px;">
        <span style="color:#f5a623;font-size:14px;">★</span>
        <span style="font-weight:600;font-size:14px;color:#888;"> {rating}</span>
        <span style="color:#888;font-size:13px;"> ({rc_str} reviews)</span>
      </div>
      <div style="margin-top:8px;">
        <a href="{link}" target="_blank"
           style="font-size:13px;color:#0066c0;text-decoration:none;font-weight:600;">
          View on Amazon
        </a>
      </div>
    </div>""", unsafe_allow_html=True)

def kpi_row(df):
    metrics = [
        ("🛍️ Total Products", f"{len(df):,}"),
        ("📂 Categories",      str(df["main_category"].nunique())),
        ("⭐ Avg Rating",      f"{df['rating'].mean():.2f}"),
        ("💸 Avg Discount",    f"{df['discount_pct'].mean():.1f}%"),
        ("💰 Avg Savings",     f"₹{df['savings'].mean():,.0f}"),
    ]
    for col, (lbl, val) in zip(st.columns(5), metrics):
        col.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def main():
    st.title("Amazon Product Recommender")
    st.caption("Content-based filtering powered by TF-IDF cosine similarity")
    st.divider()

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            "Neither `amazon_clean.csv` nor `amazon.csv` found. "
            "Run the EDA notebook to generate `amazon_clean.csv` and place it alongside `app.py`."
        )
        st.stop()

    tfidf = build_tfidf_matrix(df)

    kpi_row(df)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Choose a recommendation mode",
            ["Search & Similar", "Category Explorer", "Top Rated"],
        )
        st.divider()
        st.subheader("Filters")
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 3.5, 0.1)
        cat_filter = st.selectbox(
            "Category",
            ["All"] + sorted(df["main_category"].unique().tolist()),
        )
        st.divider()
        st.caption("Ishan Ravishankar | 23FE10CSE00641")

    display_df = df[df["rating"] >= min_rating]
    if cat_filter != "All":
        display_df = display_df[display_df["main_category"] == cat_filter]

    # RECOMMENDATION MODES

    if mode == "Search & Similar":
        st.subheader("Search products and find similar items")

        query = st.text_input(
            "Search by product name",
            placeholder="e.g. USB cable, Smart TV, Bluetooth earphones…",
        )
        top_n = st.slider("Number of recommendations", 5, 20, 10)

        if query:
            hits = display_df[
                display_df["product_name"].str.contains(query, case=False, na=False)
            ].head(30)

            if hits.empty:
                st.warning("No products matched. Try a different keyword.")
            else:
                st.success(f"Found **{len(hits)}** product(s). Select one to see recommendations.")
                selected = st.selectbox("Select a product", options=hits["product_name"].tolist())

                if selected:
                    sel_idx = df[df["product_name"] == selected].index[0]

                    st.markdown("**Selected product**")
                    render_product_card(df.iloc[sel_idx], idx=0)

                    st.markdown(f"**Top {top_n} similar products**")
                    recs = content_based_recommend(df, tfidf, sel_idx, top_n=top_n)
                    for i, (_, row) in enumerate(recs.iterrows()):
                        render_product_card(row, idx=i + 1, show_sim=True, sim_val=row["similarity"])

    elif mode == "Category Explorer":
        st.subheader("Browse top products in a category")

        c1, c2, c3 = st.columns(3)
        with c1:
            chosen_cat = st.selectbox("Category", sorted(df["main_category"].unique()))
        with c2:
            sort_by = st.selectbox("Sort by", {
                "value_score":        "Value Score",
                "rating":             "Rating",
                "rating_count_clean": "Review Count",
                "savings":            "Savings",
                "discount_pct":       "Discount %",
            })
        with c3:
            top_n_cat = st.slider("Show top N", 5, 20, 10, key="cat_n")

        cat_data = df[df["main_category"] == chosen_cat]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Products",     f"{len(cat_data):,}")
        m2.metric("Avg Rating",   f"{cat_data['rating'].mean():.2f}")
        m3.metric("Avg Discount", f"{cat_data['discount_pct'].mean():.1f}%")
        m4.metric("Avg Price",    f"₹{cat_data['discounted_price'].mean():,.0f}")

        st.divider()
        for i, (_, row) in enumerate(
            category_top_products(df, chosen_cat, sort_by=sort_by, top_n=top_n_cat).iterrows()
        ):
            render_product_card(row, idx=i + 1)

    elif mode == "Top Rated":
        st.subheader("Leaderboard")

        c1, c2 = st.columns(2)
        with c1:
            top_metric = st.radio(
                "Rank by",
                ["Value Score", "Rating", "Most Reviewed", "Highest Savings"],
                horizontal=True,
            )
        with c2:
            top_cat = st.selectbox(
                "Category",
                ["All"] + sorted(df["main_category"].unique()),
                key="tcat",
            )
        top_n_r = st.slider("Show top N", 5, 30, 15, key="top_n")

        metric_map = {
            "Value Score":     "value_score",
            "Rating":          "rating",
            "Most Reviewed":   "rating_count_clean",
            "Highest Savings": "savings",
        }
        col  = metric_map[top_metric]
        base = df if top_cat == "All" else df[df["main_category"] == top_cat]
        if col in ("value_score", "rating"):
            base = base[base["rating_count_clean"].fillna(0) >= 50]

        results = base.nlargest(top_n_r, col)
        if results.empty:
            st.info("No products found with the current filters.")
        else:
            for i, (_, row) in enumerate(results.iterrows()):
                render_product_card(row, idx=i + 1)

    st.divider()
    st.caption("ML Lab Project | Amazon Product Recommendation System")


if __name__ == "__main__":
    main()
