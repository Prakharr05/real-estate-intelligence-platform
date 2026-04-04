"""
train.py — Property Valuation Model Trainer
============================================
Trains one XGBoost model per category (plots, builder_floors, societies).
Target: price_per_sqft (fair rate for a given set of features)
Output: saves 3 model files + 3 encoder files to /models/ directory

Run: python train.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import SessionLocal
from models import Society, BuilderFloor, Plot

# ── Output directory ──────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Conservative XGBoost hyperparameters (small dataset safe) ─
XGBOOST_PARAMS = {
    "n_estimators":     100,
    "max_depth":        3,       # shallow trees → less overfit
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,       # requires 5+ samples per leaf
    "reg_alpha":        0.1,     # L1 regularization
    "reg_lambda":       1.0,     # L2 regularization
    "random_state":     42,
    "objective":        "reg:squarederror",
}

# ============================================================
# DATA LOADERS
# ============================================================

def load_plots(db):
    rows = db.query(Plot).filter(
        Plot.price_per_sqft.isnot(None),
        Plot.price_per_sqft > 0,
        Plot.plot_area_sqyd.isnot(None),
        Plot.sector != "Unknown",
        Plot.connectivity_score > 0
    ).all()

    records = []
    for r in rows:
        records.append({
            "sector":             r.sector,
            "area_sqyd":          r.plot_area_sqyd,
            "connectivity_score": r.connectivity_score,
            "is_corner":          1 if r.is_corner_plot == "Yes" else 0,
            "price_per_sqft":     r.price_per_sqft,   # target (₹/sqyd stored here)
        })
    return pd.DataFrame(records)


def load_floors(db):
    rows = db.query(BuilderFloor).filter(
        BuilderFloor.price_per_sqft.isnot(None),
        BuilderFloor.price_per_sqft > 0,
        BuilderFloor.area_sqft.isnot(None),
        BuilderFloor.bhk_type.isnot(None),
        BuilderFloor.sector != "Unknown",
        BuilderFloor.connectivity_score > 0
    ).all()

    records = []
    for r in rows:
        try:
            bhk = float(str(r.bhk_type).strip())
        except:
            continue
        records.append({
            "sector":             r.sector,
            "area_sqft":          r.area_sqft,
            "bhk_type":           bhk,
            "connectivity_score": r.connectivity_score,
            "price_per_sqft":     r.price_per_sqft,
        })
    return pd.DataFrame(records)


def load_societies(db):
    rows = db.query(Society).filter(
        Society.price_per_sqft.isnot(None),
        Society.price_per_sqft > 0,
        Society.area_sqft.isnot(None),
        Society.bhk_type.isnot(None),
        Society.sector != "Unknown",
        Society.connectivity_score > 0
    ).all()

    records = []
    for r in rows:
        try:
            bhk = float(str(r.bhk_type).strip())
        except:
            continue
        records.append({
            "sector":             r.sector,
            "area_sqft":          r.area_sqft,
            "bhk_type":           bhk,
            "connectivity_score": r.connectivity_score,
            "possession":         1 if r.possession_status == "Ready" else 0,
            "price_per_sqft":     r.price_per_sqft,
        })
    return pd.DataFrame(records)

# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_model(df, feature_cols, target_col, category_name):
    print(f"\n{'='*55}")
    print(f"  Training: {category_name.upper()} ({len(df)} rows)")
    print(f"{'='*55}")

    if len(df) < 20:
        print(f"  ⚠️ Too few rows ({len(df)}) — skipping {category_name}.")
        return None, None

    # ── Encode sector ─────────────────────────────────────────
    le = LabelEncoder()
    df = df.copy()
    df['sector_enc'] = le.fit_transform(df['sector'])

    # Final feature set (sector_enc replaces sector string)
    enc_features = [c for c in feature_cols if c != 'sector'] + ['sector_enc']

    X = df[enc_features].values
    y = df[target_col].values

    # ── Cross-validation (KFold, conservative for small data) ─
    model = XGBRegressor(**XGBOOST_PARAMS)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mae_scores, r2_scores = [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        fold_model = XGBRegressor(**XGBOOST_PARAMS)
        fold_model.fit(X_train, y_train)
        y_pred_val = fold_model.predict(X_val)
        
        mae_scores.append(mean_absolute_error(y_val, y_pred_val))
        r2_scores.append(r2_score(y_val, y_pred_val))

    mae_scores = np.array(mae_scores)
    r2_scores  = np.array(r2_scores)
    print(f"  CV MAE  : ₹{mae_scores.mean():,.0f} ± ₹{mae_scores.std():,.0f}")
    print(f"  CV R²   : {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

    # Final fit on full data
    model.fit(X, y)
    y_pred = model.predict(X)
    train_mae = mean_absolute_error(y, y_pred)
    train_r2  = r2_score(y, y_pred)
    print(f"  Train MAE: ₹{train_mae:,.0f}")
    print(f"  Train R² : {train_r2:.3f}")

    # ── Feature importance ────────────────────────────────────
    importance = dict(zip(enc_features, model.feature_importances_))
    print("  Feature importances:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:<25} {imp:.3f}")

    # ── Save model + encoder ──────────────────────────────────
    model_path   = os.path.join(MODELS_DIR, f"{category_name}_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, f"{category_name}_encoder.pkl")

    with open(model_path,   'wb') as f: pickle.dump(model, f)
    with open(encoder_path, 'wb') as f: pickle.dump(le,    f)

    # ── Save sector averages for benchmark comparison ─────────
    sector_avg = df.groupby('sector')[target_col].mean().round(2).to_dict()
    avg_path = os.path.join(MODELS_DIR, f"{category_name}_sector_avg.pkl")
    with open(avg_path, 'wb') as f: pickle.dump(sector_avg, f)

    print(f"  ✅ Saved: {model_path}")
    print(f"  ✅ Saved: {encoder_path}")
    print(f"  ✅ Saved: {avg_path}")

    return model, le


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    db = SessionLocal()

    try:
        # --- PLOTS ---
        df_plots = load_plots(db)
        if not df_plots.empty:
            train_model(
                df=df_plots,
                feature_cols=['sector', 'area_sqyd', 'connectivity_score', 'is_corner'],
                target_col='price_per_sqft',
                category_name='plots'
            )
        else:
            print("⚠️ No valid plot data found.")

        # --- BUILDER FLOORS ---
        df_floors = load_floors(db)
        if not df_floors.empty:
            train_model(
                df=df_floors,
                feature_cols=['sector', 'area_sqft', 'bhk_type', 'connectivity_score'],
                target_col='price_per_sqft',
                category_name='floors'
            )
        else:
            print("⚠️ No valid floor data found.")

        # --- SOCIETIES ---
        df_societies = load_societies(db)
        if not df_societies.empty:
            train_model(
                df=df_societies,
                feature_cols=['sector', 'area_sqft', 'bhk_type',
                              'connectivity_score', 'possession'],
                target_col='price_per_sqft',
                category_name='societies'
            )
        else:
            print("⚠️ No valid society data found.")

    finally:
        db.close()

    print("\n🚀 All models trained and saved to /models/")
    print("   Run app.py to serve predictions in the UI.")