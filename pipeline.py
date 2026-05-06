import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── 1. CHARGEMENT ──────────────────────────────────────────────────────────────

train = pd.read_csv("airbnb_train.csv")
test  = pd.read_csv("airbnb_test.csv", index_col=0)
test.index.name = "id"
test = test.reset_index()

print(f"Train : {train.shape}  |  Test : {test.shape}")
print(f"Target — mean: {train['log_price'].mean():.3f}  std: {train['log_price'].std():.3f}")

# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────────

# Amenities les plus fréquentes à extraire comme features binaires
TOP_AMENITIES = [
    "TV", "Wireless Internet", "Air conditioning", "Kitchen", "Heating",
    "Washer", "Dryer", "Elevator in building", "Gym", "Pool",
    "Free parking on premises", "Breakfast", "Pets allowed",
    "Smoke detector", "Carbon monoxide detector", "Doorman",
    "Laptop friendly workspace", "Family/kid friendly", "Self Check-In",
]

def parse_amenities(series):
    rows = []
    for val in series:
        if pd.isna(val):
            rows.append({a: 0 for a in TOP_AMENITIES})
            continue
        text = val.lower()
        row = {a: int(a.lower() in text) for a in TOP_AMENITIES}
        row["n_amenities"] = len(re.findall(r'"[^"]*"|\b\w+\b', val))
        rows.append(row)
    return pd.DataFrame(rows, index=series.index)

def feature_engineer(df):
    df = df.copy()

    # Dates → ancienneté en jours (par rapport à une date de référence fixe)
    ref = pd.Timestamp("2017-10-01")
    for col in ["host_since", "first_review", "last_review"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_days"] = (ref - parsed).dt.days
        df.drop(columns=[col], inplace=True)

    # host_response_rate : "100%" → 100.0
    df["host_response_rate"] = (
        df["host_response_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )

    # cleaning_fee booléen → binaire
    df["cleaning_fee"] = df["cleaning_fee"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)
    df["instant_bookable"]        = df["instant_bookable"].map({"t": 1, "f": 0}).fillna(0)
    df["host_has_profile_pic"]    = df["host_has_profile_pic"].map({"t": 1, "f": 0}).fillna(0)
    df["host_identity_verified"]  = df["host_identity_verified"].map({"t": 1, "f": 0}).fillna(0)

    # Longueur de la description
    df["desc_len"] = df["description"].fillna("").apply(len)
    df["name_len"] = df["name"].fillna("").apply(len)
    df.drop(columns=["description", "name"], inplace=True)

    # Amenities
    amen = parse_amenities(df["amenities"])
    df = pd.concat([df.drop(columns=["amenities"]), amen], axis=1)

    return df

train_eng = feature_engineer(train.drop(columns=["log_price"]))
train_eng["log_price"] = train["log_price"].values
test_eng  = feature_engineer(test)

# ── 3. COLONNES PAR TYPE ───────────────────────────────────────────────────────

TARGET = "log_price"
ID_COL = "id"

CAT_COLS = ["property_type", "room_type", "bed_type",
            "cancellation_policy", "city", "neighbourhood", "zipcode"]

NUM_COLS = [c for c in train_eng.columns
            if c not in CAT_COLS + [TARGET, ID_COL]]

print(f"\nFeatures numériques : {len(NUM_COLS)}")
print(f"Features catégorielles : {len(CAT_COLS)}")

# ── 4. PRÉPROCESSEUR ───────────────────────────────────────────────────────────

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM_COLS),
    ("cat", cat_pipe, CAT_COLS),
], remainder="drop")

X_train = train_eng.drop(columns=[TARGET, ID_COL])
y_train = train_eng[TARGET]
X_test  = test_eng.drop(columns=[ID_COL])

# ── 5. MODÈLES ─────────────────────────────────────────────────────────────────

def rmse_cv(model_pipeline, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model_pipeline, X, y,
                             scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    return -scores.mean(), scores.std()

models = {
    "Ridge": Pipeline([
        ("pre", preprocessor),
        ("model", Ridge(alpha=10)),
    ]),
    "LightGBM": Pipeline([
        ("pre", preprocessor),
        ("model", lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )),
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("model", xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method="hist", n_jobs=-1, verbosity=0,
        )),
    ]),
}

print("\n-- Validation croisee (5-fold RMSE) --")
results = {}
for name, pipe in models.items():
    mean, std = rmse_cv(pipe, X_train, y_train)
    results[name] = mean
    print(f"  {name:<12}  RMSE = {mean:.4f} ± {std:.4f}")

# ── 6. ENTRAÎNEMENT FINAL + PRÉDICTIONS ────────────────────────────────────────

best_name = min(results, key=results.get)
print(f"\nMeilleur modèle : {best_name} (RMSE = {results[best_name]:.4f})")

best_pipe = models[best_name]
best_pipe.fit(X_train, y_train)

preds = best_pipe.predict(X_test)

submission = pd.DataFrame({"id": test_eng[ID_COL], "logpred": preds})
submission.to_csv("predictions.csv", index=False)
print(f"\nPredictions sauvegardees -> predictions.csv  ({len(submission)} lignes)")
print(submission.head())
