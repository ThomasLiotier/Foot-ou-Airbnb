"""
Prédiction des résultats de la Ligue 1 2025-2026.

Pipeline:
  1. Chargement et filtrage des matchs de Ligue 1 (2013-2024)
  2. Calcul d'un classement ELO mis à jour match par match
  3. Features par équipe: forme récente, ELO, valeur marchande, stats club
  4. Validation temporelle (train < 2023, val 2023-2024)
  5. Modèle final ré-entraîné sur tout l'historique, prédiction 2025-2026

Sortie:
  - predictions.csv au format `game_id,results` (1, 0, -1)
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------------------

def load_data() -> dict[str, pd.DataFrame]:
    matches_hist = pd.read_csv(DATA_DIR / "matchs_2013_2024.csv")
    matches_2025 = pd.read_csv(DATA_DIR / "match_2025.csv")
    clubs = pd.read_csv(DATA_DIR / "clubs_fr.csv")
    valuations = pd.read_csv(DATA_DIR / "player_valuation_before_season.csv")

    # Garder uniquement les matchs de championnat (Ligue 1)
    matches_hist = matches_hist[matches_hist["competition_type"] == "domestic_league"].copy()
    matches_hist["date"] = pd.to_datetime(matches_hist["date"])
    matches_2025["date"] = pd.to_datetime(matches_2025["date"])
    valuations["date"] = pd.to_datetime(valuations["date"])

    # Trier chronologiquement pour les calculs incrémentaux (ELO, forme)
    matches_hist = matches_hist.sort_values("date").reset_index(drop=True)
    matches_2025 = matches_2025.sort_values("date").reset_index(drop=True)

    return {
        "hist": matches_hist,
        "future": matches_2025,
        "clubs": clubs,
        "valuations": valuations,
    }


# ---------------------------------------------------------------------------
# 2. ELO incrémental
# ---------------------------------------------------------------------------

def compute_elo_features(matches: pd.DataFrame, k: float = 20.0, home_adv: float = 80.0) -> pd.DataFrame:
    """Calcule l'ELO de chaque équipe AVANT chaque match (pas de fuite de données)."""
    elo = defaultdict(lambda: 1500.0)
    home_elos, away_elos = [], []

    for row in matches.itertuples(index=False):
        h, a = row.home_club_id, row.away_club_id
        eh, ea = elo[h], elo[a]
        home_elos.append(eh)
        away_elos.append(ea)

        # Probabilité attendue avec avantage du domicile
        exp_home = 1.0 / (1.0 + 10 ** (-(eh + home_adv - ea) / 400))
        # Score réel: 1 = victoire dom, 0.5 = nul, 0 = défaite dom
        if row.results == 1:
            score_home = 1.0
        elif row.results == 0:
            score_home = 0.5
        else:
            score_home = 0.0

        elo[h] = eh + k * (score_home - exp_home)
        elo[a] = ea + k * ((1 - score_home) - (1 - exp_home))

    return pd.DataFrame({"home_elo": home_elos, "away_elo": away_elos}), dict(elo)


def attach_future_elo(future: pd.DataFrame, elo_state: dict) -> pd.DataFrame:
    """Pour les matchs futurs, on prend l'ELO courant à la fin de l'historique."""
    future = future.copy()
    future["home_elo"] = future["home_club_id"].map(lambda c: elo_state.get(c, 1500.0))
    future["away_elo"] = future["away_club_id"].map(lambda c: elo_state.get(c, 1500.0))
    return future


# ---------------------------------------------------------------------------
# 3. Features de forme récente (rolling, sans fuite)
# ---------------------------------------------------------------------------

def compute_form_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Pour chaque match, calcule pour chaque équipe (domicile/extérieur) :
      - points moyens sur les `window` derniers matchs
      - différence de buts moyenne
      - buts marqués / encaissés
    en n'utilisant QUE les matchs antérieurs.
    """
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    rows = []

    for row in matches.itertuples(index=False):
        h, a = row.home_club_id, row.away_club_id
        feats = {}
        for side, club in [("home", h), ("away", a)]:
            past = list(history[club])
            if past:
                pts = np.mean([p["pts"] for p in past])
                gf = np.mean([p["gf"] for p in past])
                ga = np.mean([p["ga"] for p in past])
                gd = gf - ga
                n = len(past)
            else:
                pts, gf, ga, gd, n = np.nan, np.nan, np.nan, np.nan, 0
            feats[f"{side}_form_pts"] = pts
            feats[f"{side}_form_gf"] = gf
            feats[f"{side}_form_ga"] = ga
            feats[f"{side}_form_gd"] = gd
            feats[f"{side}_form_n"] = n
        rows.append(feats)

        # Mise à jour de l'historique APRÈS calcul
        if row.results == 1:
            pts_h, pts_a = 3, 0
        elif row.results == 0:
            pts_h, pts_a = 1, 1
        else:
            pts_h, pts_a = 0, 3
        history[h].append({"pts": pts_h, "gf": row.home_club_goals, "ga": row.away_club_goals})
        history[a].append({"pts": pts_a, "gf": row.away_club_goals, "ga": row.home_club_goals})

    return pd.DataFrame(rows), history


def attach_future_form(future: pd.DataFrame, history: dict, window: int = 5) -> pd.DataFrame:
    """Pour les matchs futurs, on prend la forme finale de chaque équipe."""
    rows = []
    for row in future.itertuples(index=False):
        feats = {}
        for side, club in [("home", row.home_club_id), ("away", row.away_club_id)]:
            past = list(history.get(club, []))
            if past:
                pts = np.mean([p["pts"] for p in past])
                gf = np.mean([p["gf"] for p in past])
                ga = np.mean([p["ga"] for p in past])
                feats[f"{side}_form_pts"] = pts
                feats[f"{side}_form_gf"] = gf
                feats[f"{side}_form_ga"] = ga
                feats[f"{side}_form_gd"] = gf - ga
                feats[f"{side}_form_n"] = len(past)
            else:
                feats[f"{side}_form_pts"] = np.nan
                feats[f"{side}_form_gf"] = np.nan
                feats[f"{side}_form_ga"] = np.nan
                feats[f"{side}_form_gd"] = np.nan
                feats[f"{side}_form_n"] = 0
        rows.append(feats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Head-to-head historique
# ---------------------------------------------------------------------------

def compute_h2h_features(matches: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, dict]:
    history: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=window))
    rows = []

    for row in matches.itertuples(index=False):
        key = tuple(sorted([row.home_club_id, row.away_club_id]))
        past = list(history[key])
        if past:
            # Score H2H normalisé du point de vue de l'équipe à domicile actuelle
            home_wins = sum(1 for p in past if p["winner"] == row.home_club_id)
            away_wins = sum(1 for p in past if p["winner"] == row.away_club_id)
            draws = sum(1 for p in past if p["winner"] == 0)
            n = len(past)
            rows.append({"h2h_home_winrate": home_wins / n, "h2h_draw_rate": draws / n, "h2h_n": n})
        else:
            rows.append({"h2h_home_winrate": np.nan, "h2h_draw_rate": np.nan, "h2h_n": 0})

        if row.results == 1:
            winner = row.home_club_id
        elif row.results == -1:
            winner = row.away_club_id
        else:
            winner = 0
        history[key].append({"winner": winner})

    return pd.DataFrame(rows), history


def attach_future_h2h(future: pd.DataFrame, history: dict) -> pd.DataFrame:
    rows = []
    for row in future.itertuples(index=False):
        key = tuple(sorted([row.home_club_id, row.away_club_id]))
        past = list(history.get(key, []))
        if past:
            home_wins = sum(1 for p in past if p["winner"] == row.home_club_id)
            away_wins = sum(1 for p in past if p["winner"] == row.away_club_id)
            draws = sum(1 for p in past if p["winner"] == 0)
            n = len(past)
            rows.append({"h2h_home_winrate": home_wins / n, "h2h_draw_rate": draws / n, "h2h_n": n})
        else:
            rows.append({"h2h_home_winrate": np.nan, "h2h_draw_rate": np.nan, "h2h_n": 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Valeur marchande de l'équipe (avant chaque saison)
# ---------------------------------------------------------------------------

def compute_squad_values(valuations: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque (club, année), somme la valeur du squad la plus proche AVANT le 1er août."""
    valuations = valuations.copy()
    valuations["year"] = valuations["date"].dt.year
    valuations["month"] = valuations["date"].dt.month

    # Pour chaque saison N (qui commence en août N), on prend les valuations de l'année N
    # antérieures à août. À défaut, la dernière valuation connue de l'année N-1.
    rows = []
    for (club, year), grp in valuations.groupby(["current_club_id", "year"]):
        # Pour chaque joueur on prend sa dernière valeur connue dans l'année
        latest_per_player = grp.sort_values("date").groupby("player_id").tail(1)
        rows.append({
            "club_id": club,
            "season": year,
            "squad_value": latest_per_player["market_value_in_eur"].sum(),
            "squad_top3_value": latest_per_player["market_value_in_eur"].nlargest(3).sum(),
            "squad_n_players": len(latest_per_player),
        })
    return pd.DataFrame(rows)


def attach_squad_values(matches: pd.DataFrame, squad_vals: pd.DataFrame) -> pd.DataFrame:
    """Joint la valeur du squad pour la saison correspondant à chaque match.

    Convention: une saison `season` couvre août `season` -> mai `season+1`.
    On rapproche la valuation de l'année calendaire de la saison.
    """
    out = matches.copy()
    if "season" not in out.columns:
        # Pour les matchs futurs, déduire la saison depuis la date (année calendaire - 1 si avant août)
        m = out["date"].dt.month
        out["season"] = np.where(m >= 7, out["date"].dt.year, out["date"].dt.year - 1)

    sv = squad_vals.rename(columns={"club_id": "home_club_id", "season": "season"})
    out = out.merge(
        sv.rename(columns={
            "squad_value": "home_squad_value",
            "squad_top3_value": "home_squad_top3",
            "squad_n_players": "home_squad_n",
        }),
        on=["home_club_id", "season"], how="left",
    )
    sv2 = squad_vals.rename(columns={"club_id": "away_club_id"})
    out = out.merge(
        sv2.rename(columns={
            "squad_value": "away_squad_value",
            "squad_top3_value": "away_squad_top3",
            "squad_n_players": "away_squad_n",
        }),
        on=["away_club_id", "season"], how="left",
    )
    return out


# ---------------------------------------------------------------------------
# 6. Stats club (clubs_fr.csv) — utilisées surtout pour 2025
# ---------------------------------------------------------------------------

_NET_TRANSFER_RE = re.compile(r"([+-]?)€?(-?\d+(?:\.\d+)?)([mk]?)", re.IGNORECASE)


def parse_transfer(s) -> float:
    if pd.isna(s):
        return np.nan
    s = str(s).replace(" ", "")
    m = _NET_TRANSFER_RE.search(s)
    if not m:
        return np.nan
    sign, num, unit = m.groups()
    val = float(num)
    if unit.lower() == "m":
        val *= 1_000_000
    elif unit.lower() == "k":
        val *= 1_000
    if sign == "-" or s.startswith("€-") or s.startswith("-"):
        val = -abs(val)
    return val


def attach_club_stats(matches: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    c = clubs.copy()
    c["net_transfer"] = c["net_transfer_record"].apply(parse_transfer)
    keep = ["club_id", "squad_size", "average_age", "foreigners_percentage",
            "national_team_players", "stadium_seats", "net_transfer"]
    c = c[keep]

    out = matches.merge(
        c.add_prefix("home_").rename(columns={"home_club_id": "home_club_id"}),
        on="home_club_id", how="left",
    )
    out = out.merge(
        c.add_prefix("away_").rename(columns={"away_club_id": "away_club_id"}),
        on="away_club_id", how="left",
    )
    return out


# ---------------------------------------------------------------------------
# 7. Construction du dataset final
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "elo_diff", "home_elo", "away_elo",
    "home_form_pts", "away_form_pts",
    "home_form_gd", "away_form_gd",
    "home_form_gf", "away_form_gf",
    "home_form_ga", "away_form_ga",
    "h2h_home_winrate", "h2h_draw_rate", "h2h_n",
    "squad_value_diff", "squad_top3_diff",
    "home_squad_value", "away_squad_value",
    "home_squad_size", "away_squad_size",
    "home_average_age", "away_average_age",
    "home_foreigners_percentage", "away_foreigners_percentage",
    "home_stadium_seats",
    "home_net_transfer", "away_net_transfer",
    "rest_days_home", "rest_days_away",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["squad_value_diff"] = df.get("home_squad_value", 0).fillna(0) - df.get("away_squad_value", 0).fillna(0)
    df["squad_top3_diff"] = df.get("home_squad_top3", 0).fillna(0) - df.get("away_squad_top3", 0).fillna(0)
    return df


def add_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Jours depuis le dernier match de chaque équipe."""
    last_seen: dict[int, pd.Timestamp] = {}
    rh, ra = [], []
    for row in matches.sort_values("date").itertuples(index=False):
        d = row.date
        rh.append((d - last_seen[row.home_club_id]).days if row.home_club_id in last_seen else np.nan)
        ra.append((d - last_seen[row.away_club_id]).days if row.away_club_id in last_seen else np.nan)
        last_seen[row.home_club_id] = d
        last_seen[row.away_club_id] = d
    out = matches.sort_values("date").copy()
    out["rest_days_home"] = rh
    out["rest_days_away"] = ra
    return out.sort_index()


# ---------------------------------------------------------------------------
# 8. Pipeline principal
# ---------------------------------------------------------------------------

def build_features():
    data = load_data()
    hist, future = data["hist"], data["future"]
    clubs, valuations = data["clubs"], data["valuations"]

    print(f"[i] Matchs historiques (Ligue 1 / domestic_league): {len(hist)}")
    print(f"[i] Matchs à prédire (saison 2025-2026): {len(future)}")
    print(f"[i] Distribution des résultats historiques :\n{hist['results'].value_counts(normalize=True).round(3)}")

    # ELO
    elo_df, elo_state = compute_elo_features(hist)
    hist = pd.concat([hist.reset_index(drop=True), elo_df.reset_index(drop=True)], axis=1)
    future = attach_future_elo(future, elo_state)

    # Forme récente
    form_df, form_state = compute_form_features(hist, window=5)
    hist = pd.concat([hist.reset_index(drop=True), form_df.reset_index(drop=True)], axis=1)
    future_form = attach_future_form(future, form_state, window=5)
    future = pd.concat([future.reset_index(drop=True), future_form.reset_index(drop=True)], axis=1)

    # Head-to-head
    h2h_df, h2h_state = compute_h2h_features(hist, window=5)
    hist = pd.concat([hist.reset_index(drop=True), h2h_df.reset_index(drop=True)], axis=1)
    future_h2h = attach_future_h2h(future, h2h_state)
    future = pd.concat([future.reset_index(drop=True), future_h2h.reset_index(drop=True)], axis=1)

    # Valeur du squad
    squad_vals = compute_squad_values(valuations)
    hist = attach_squad_values(hist, squad_vals)
    future = attach_squad_values(future, squad_vals)

    # Stats club statiques (utiles pour les promus en 2025)
    hist = attach_club_stats(hist, clubs)
    future = attach_club_stats(future, clubs)

    # Repos
    hist = add_rest_days(hist)
    future = add_rest_days(future)

    # Features dérivées
    hist = add_derived_features(hist)
    future = add_derived_features(future)

    return hist, future


# ---------------------------------------------------------------------------
# 9. Entraînement et évaluation
# ---------------------------------------------------------------------------

def _fit_logreg(X_train, y_train, medians, C=0.3):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, C=C)),
    ])
    pipe.fit(X_train.fillna(medians), y_train)
    return pipe


def _fit_hgbm(X_train, y_train, **kwargs):
    params = dict(
        max_iter=600, learning_rate=0.03, max_depth=4,
        min_samples_leaf=80, l2_regularization=2.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        random_state=42,
    )
    params.update(kwargs)
    m = HistGradientBoostingClassifier(**params)
    m.fit(X_train, y_train)
    return m


def train_and_evaluate(hist: pd.DataFrame):
    df = hist.dropna(subset=["results"]).copy()
    df = df.dropna(subset=["home_form_pts", "away_form_pts"])

    train = df[df["season"] < 2023]
    val = df[df["season"] >= 2023]
    print(f"[i] Train: {len(train)} matchs (saisons {train.season.min()}-{train.season.max()})")
    print(f"[i] Val  : {len(val)} matchs (saisons {val.season.min()}-{val.season.max()})")

    X_train, y_train = train[FEATURE_COLUMNS], train["results"].astype(int)
    X_val, y_val = val[FEATURE_COLUMNS], val["results"].astype(int)
    medians = X_train.median()

    # Baseline: toujours prédire victoire à domicile (la classe majoritaire)
    base_acc = (y_val == 1).mean()
    print(f"[baseline] Toujours 'victoire domicile' -> accuracy {base_acc:.3f}")

    # Modèle 1: régression logistique régularisée
    logreg = _fit_logreg(X_train, y_train, medians, C=0.3)
    pred_lr = logreg.predict(X_val.fillna(medians))
    proba_lr = logreg.predict_proba(X_val.fillna(medians))
    acc_lr = accuracy_score(y_val, pred_lr)
    ll_lr = log_loss(y_val, proba_lr, labels=logreg.classes_)
    print(f"[logreg]   accuracy={acc_lr:.3f}  log_loss={ll_lr:.3f}")

    # Modèle 2: gradient boosting (régularisé + early stopping)
    gbm = _fit_hgbm(X_train, y_train)
    pred_gbm = gbm.predict(X_val)
    proba_gbm = gbm.predict_proba(X_val)
    acc_gbm = accuracy_score(y_val, pred_gbm)
    ll_gbm = log_loss(y_val, proba_gbm, labels=gbm.classes_)
    print(f"[hgbm]     accuracy={acc_gbm:.3f}  log_loss={ll_gbm:.3f}  (n_iter={gbm.n_iter_})")

    # Modèle 3: ensemble (moyenne des probas) — souvent plus robuste
    # Aligner l'ordre des classes (sklearn les trie -1, 0, 1)
    assert list(logreg.classes_) == list(gbm.classes_)
    proba_ens = 0.5 * proba_lr + 0.5 * proba_gbm
    pred_ens = gbm.classes_[proba_ens.argmax(axis=1)]
    acc_ens = accuracy_score(y_val, pred_ens)
    ll_ens = log_loss(y_val, proba_ens, labels=gbm.classes_)
    print(f"[ensemble] accuracy={acc_ens:.3f}  log_loss={ll_ens:.3f}")

    # Choix du meilleur modèle (par log_loss, plus robuste que l'accuracy seule)
    candidates = [
        ("logreg", logreg, ll_lr, acc_lr),
        ("hgbm", gbm, ll_gbm, acc_gbm),
        ("ensemble", ("ensemble", logreg, gbm), ll_ens, acc_ens),
    ]
    best_name, best_obj, best_ll, best_acc = min(candidates, key=lambda x: x[2])
    print(f"\n[choix] Meilleur modèle (log_loss): {best_name}  (acc={best_acc:.3f}, ll={best_ll:.3f})")

    print(f"\n[{best_name}] rapport détaillé:")
    if best_name == "logreg":
        best_pred = pred_lr
    elif best_name == "hgbm":
        best_pred = pred_gbm
    else:
        best_pred = pred_ens
    print(classification_report(y_val, best_pred, digits=3))

    # Importance des features (sur HGBM, plus interprétable)
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(gbm, X_val, y_val, n_repeats=5, random_state=42, n_jobs=1)
        imp = pd.Series(r.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print("\n[hgbm] importance par permutation (top 10):")
        print(imp.head(10).round(4))
    except Exception as e:
        print(f"[!] impossible de calculer l'importance: {e}")

    return best_name, best_obj, medians


# ---------------------------------------------------------------------------
# 10. Prédiction finale
# ---------------------------------------------------------------------------

def predict_future(hist: pd.DataFrame, future: pd.DataFrame, best_name: str, best_obj, medians):
    df = hist.dropna(subset=["results", "home_form_pts", "away_form_pts"]).copy()
    X = df[FEATURE_COLUMNS]
    y = df["results"].astype(int)

    # Ré-entraînement sur TOUT l'historique pour la prédiction finale
    if best_name == "logreg":
        model = _fit_logreg(X, y, medians, C=0.3)
        X_future = future[FEATURE_COLUMNS].fillna(medians)
        proba = model.predict_proba(X_future)
        classes = model.classes_
    elif best_name == "hgbm":
        model = _fit_hgbm(X, y)
        X_future = future[FEATURE_COLUMNS]
        proba = model.predict_proba(X_future)
        classes = model.classes_
    else:  # ensemble
        lr = _fit_logreg(X, y, medians, C=0.3)
        gbm = _fit_hgbm(X, y)
        X_future_imp = future[FEATURE_COLUMNS].fillna(medians)
        proba_lr = lr.predict_proba(X_future_imp)
        proba_gbm = gbm.predict_proba(future[FEATURE_COLUMNS])
        proba = 0.5 * proba_lr + 0.5 * proba_gbm
        classes = gbm.classes_

    preds = classes[proba.argmax(axis=1)]

    out = pd.DataFrame({
        "game_id": future["game_id"].values,
        "results": preds.astype(int),
    })
    out_path = DATA_DIR / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[i] Prédictions écrites dans {out_path}  (modèle utilisé: {best_name})")
    print(f"[i] Distribution prédite:\n{out['results'].value_counts(normalize=True).round(3)}")

    # Sauvegarde aussi des probabilités pour analyse
    proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    proba_df.insert(0, "game_id", future["game_id"].values)
    proba_df.to_csv(DATA_DIR / "predictions_proba.csv", index=False)
    print(f"[i] Probabilités écrites dans {DATA_DIR / 'predictions_proba.csv'}")
    return out


def main():
    hist, future = build_features()
    best_name, best_obj, medians = train_and_evaluate(hist)
    predict_future(hist, future, best_name, best_obj, medians)


if __name__ == "__main__":
    main()
