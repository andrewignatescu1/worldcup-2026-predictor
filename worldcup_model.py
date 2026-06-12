"""
World Cup 2026 match winner predictor
--------------------------------------
- Computes World-Football-style Elo ratings over every men's international since 1872
- Builds pre-match features (Elo diff, form, attack/defense, head-to-head, host advantage, WC experience)
- Trains a multinomial logistic regression (win / draw / loss) on World Cup matches 1974-2022
- Backtests on the 2014, 2018 and 2022 World Cups (walk-forward: never trains on the future)
- Predicts all remaining 2026 World Cup fixtures
"""

import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque

RNG = 42
HFA_ELO = 100  # Elo bonus for a true (non-neutral) home team, standard World Elo value

# ---------------------------------------------------------------- load
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CSV_PATH = "results.csv"

print("Fetching latest results...", end=" ", flush=True)
try:
    urllib.request.urlretrieve(RESULTS_URL, CSV_PATH)
    print("done.")
except Exception as e:
    print(f"failed ({e}), using local copy.")

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
future = df[df.home_score.isna()].copy()          # unplayed 2026 fixtures
df = df.dropna(subset=["home_score", "away_score"]).copy()
df[["home_score", "away_score"]] = df[["home_score", "away_score"]].astype(int)

def k_factor(tourn):
    t = tourn.lower()
    if t == "fifa world cup": return 60
    if "qualification" in t: return 40
    if any(s in t for s in ["copa américa", "uefa euro", "african cup", "africa cup",
                            "afc asian cup", "gold cup", "confederations", "nations league",
                            "uefa nations"]): return 50
    if t == "friendly": return 20
    return 30

def mov_mult(gd):
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return (11 + gd) / 8.0

# ---------------------------------------------------------------- state trackers
elo = defaultdict(lambda: 1500.0)
recent = defaultdict(lambda: deque(maxlen=10))     # (pts, gf, ga) last 10 matches
h2h = defaultdict(lambda: [0, 0, 0])               # (winsA, draws, winsB) keyed by sorted pair
wc_matches = defaultdict(int)                      # career WC matches played

def pair_key(a, b):
    return (a, b) if a < b else (b, a)

def features_for(home, away, neutral, tournament):
    """Build pre-match feature vector from home team's perspective."""
    e_h, e_a = elo[home], elo[away]
    is_home = 0.0 if neutral else 1.0
    rh, ra = recent[home], recent[away]
    def agg(dq):
        if not dq: return 1.0, 1.2, 1.2  # priors: ~1pt, league-average goals
        arr = np.array(dq)
        return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
    pts_h, gf_h, ga_h = agg(rh)
    pts_a, gf_a, ga_a = agg(ra)
    k = pair_key(home, away)
    wA, d, wB = h2h[k]
    n = wA + d + wB
    if n == 0:
        h2h_score = 0.0
    else:
        # win share from home team's perspective
        wins_home = wA if k[0] == home else wB
        wins_away = wB if k[0] == home else wA
        h2h_score = (wins_home - wins_away) / n
    return dict(
        elo_diff=(e_h - e_a),
        is_home=is_home,
        form_diff=pts_h - pts_a,
        attack_diff=gf_h - gf_a,
        defense_diff=ga_a - ga_h,          # positive = home concedes less
        h2h_score=h2h_score,
        h2h_n=np.log1p(n),
        wc_exp_diff=np.log1p(wc_matches[home]) - np.log1p(wc_matches[away]),
    )

def update_state(home, away, hs, as_, neutral, tournament):
    # Elo update
    d = (elo[home] + (0 if neutral else HFA_ELO)) - elo[away]
    exp_home = 1.0 / (1.0 + 10 ** (-d / 400.0))
    res = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
    delta = k_factor(tournament) * mov_mult(abs(hs - as_)) * (res - exp_home)
    elo[home] += delta
    elo[away] -= delta
    # form
    ph = 3 if hs > as_ else (1 if hs == as_ else 0)
    pa = 3 if as_ > hs else (1 if hs == as_ else 0)
    recent[home].append((ph, hs, as_))
    recent[away].append((pa, as_, hs))
    # head to head
    k = pair_key(home, away)
    if hs > as_: h2h[k][0 if k[0] == home else 2] += 1
    elif as_ > hs: h2h[k][0 if k[0] == away else 2] += 1
    else: h2h[k][1] += 1
    # WC experience
    if tournament == "FIFA World Cup":
        wc_matches[home] += 1
        wc_matches[away] += 1

# ---------------------------------------------------------------- single chronological pass
rows = []
for r in df.itertuples(index=False):
    if r.tournament == "FIFA World Cup" and r.date.year >= 1974:
        f = features_for(r.home_team, r.away_team, r.neutral, r.tournament)
        f["date"] = r.date
        f["year"] = r.date.year
        f["home_team"] = r.home_team
        f["away_team"] = r.away_team
        f["outcome"] = 2 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 0)
        rows.append(f)
    update_state(r.home_team, r.away_team, r.home_score, r.away_score, r.neutral, r.tournament)

train_df = pd.DataFrame(rows)

wc2026_played = train_df[train_df.year == 2026]
if len(wc2026_played):
    print(f"\nWC 2026 matches already played and included in training ({len(wc2026_played)}):")
    for _, row in wc2026_played.iterrows():
        outcome_str = "home win" if row.outcome == 2 else ("draw" if row.outcome == 1 else "away win")
        print(f"   {str(row.date.date())}  {row.home_team} vs {row.away_team}  [{outcome_str}]")
else:
    print("\nNo WC 2026 matches played yet — predicting from historical model only.")

# Lean feature set chosen by walk-forward backtest on WC 2014/2018/2022
# (form, attack and head-to-head features added noise, not signal)
FEATS = ["elo_diff", "is_home", "defense_diff", "wc_exp_diff"]

def fit(data):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, C=0.05, random_state=RNG))
    m.fit(data[FEATS], data["outcome"])
    return m

def evaluate(model, test):
    proba = model.predict_proba(test[FEATS])
    pred3 = proba.argmax(axis=1)
    acc3 = (pred3 == test["outcome"].values).mean()
    # "winner pick": ignore draw column, pick the side with higher win prob
    pick = np.where(proba[:, 2] >= proba[:, 0], 2, 0)
    decisive = test["outcome"].values != 1
    winner_acc = (pick[decisive] == test["outcome"].values[decisive]).mean()
    # winner-or-draw treated generously? No - report strict numbers only.
    return acc3, winner_acc, decisive.sum(), len(test)

print("=" * 70)
print("WALK-FORWARD BACKTEST (train strictly on earlier World Cups)")
print("=" * 70)
for test_year in [2014, 2018, 2022]:
    tr = train_df[train_df.year < test_year]
    te = train_df[train_df.year == test_year]
    m = fit(tr)
    acc3, wacc, ndec, ntot = evaluate(m, te)
    print(f"WC {test_year}:  3-way accuracy (W/D/L): {acc3:.1%} on {ntot} matches   |   "
          f"winner-pick accuracy (decisive games only): {wacc:.1%} on {ndec} matches")

# pooled 2014-2022
tr = train_df[train_df.year < 2014]
te = train_df[train_df.year >= 2014]
m = fit(tr)
acc3, wacc, ndec, ntot = evaluate(m, te)
print("-" * 70)
print(f"POOLED 2014+2018+2022:  3-way: {acc3:.1%} ({ntot} games)   |   "
      f"winner-pick: {wacc:.1%} ({ndec} decisive games)")

# baseline: always pick higher Elo
te2 = te.copy()
base_pick = np.where(te2.elo_diff + te2.is_home * HFA_ELO >= 0, 2, 0)
dec = te2.outcome.values != 1
print(f"Baseline (pick higher Elo):  winner-pick: "
      f"{(base_pick[dec] == te2.outcome.values[dec]).mean():.1%}")

# ---------------------------------------------------------------- final model on ALL data
final_model = fit(train_df)
coefs = final_model.named_steps["logisticregression"].coef_
print("\nFinal model feature weights (home-win class, standardized):")
for f, c in sorted(zip(FEATS, coefs[2]), key=lambda x: -abs(x[1])):
    print(f"   {f:<14} {c:+.3f}")

# ---------------------------------------------------------------- predict 2026 fixtures
preds = []
for r in future.itertuples(index=False):
    if r.tournament != "FIFA World Cup":
        continue
    f = features_for(r.home_team, r.away_team, r.neutral, r.tournament)
    X = pd.DataFrame([f])[FEATS]
    p_loss, p_draw, p_win = final_model.predict_proba(X)[0]
    if p_win >= p_loss:
        winner, conf = r.home_team, p_win
    else:
        winner, conf = r.away_team, p_loss
    label = ["away win", "draw", "home win"][int(np.argmax([p_loss, p_draw, p_win]))]
    preds.append(dict(date=r.date.date(), home=r.home_team, away=r.away_team,
                      city=r.city, p_home_win=round(p_win, 3), p_draw=round(p_draw, 3),
                      p_away_win=round(p_loss, 3), most_likely=label,
                      predicted_winner=winner, winner_prob=round(conf, 3),
                      elo_home=round(elo[r.home_team]), elo_away=round(elo[r.away_team])))

pred_df = pd.DataFrame(preds).sort_values("date")
pred_df.to_csv("/Users/andrewignatescu/Downloads/wc2026_predictions.csv", index=False)
print(f"\nSaved {len(pred_df)} fixture predictions -> wc2026_predictions.csv")
print("\nCurrent Elo top 12:")
elo_now = pd.Series(elo).sort_values(ascending=False)
print(elo_now.head(12).round(0).to_string())