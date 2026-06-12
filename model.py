import os, ssl, urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (VotingClassifier, HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
HFA_ELO = 100
RNG = 42
FEATS = ["elo_diff", "is_home", "defense_diff", "wc_exp_diff",
         "elo_momentum_diff", "streak_diff"]

WC2026_GROUPS = {
    'A': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'D': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

FLAGS = {
    'Spain': '🇪🇸', 'Argentina': '🇦🇷', 'France': '🇫🇷', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'Brazil': '🇧🇷', 'Germany': '🇩🇪', 'Portugal': '🇵🇹', 'Netherlands': '🇳🇱',
    'Belgium': '🇧🇪', 'Colombia': '🇨🇴', 'Uruguay': '🇺🇾', 'Japan': '🇯🇵',
    'Mexico': '🇲🇽', 'Ecuador': '🇪🇨', 'Morocco': '🇲🇦', 'Senegal': '🇸🇳',
    'United States': '🇺🇸', 'Canada': '🇨🇦', 'Australia': '🇦🇺', 'Switzerland': '🇨🇭',
    'Croatia': '🇭🇷', 'South Korea': '🇰🇷', 'Norway': '🇳🇴', 'Austria': '🇦🇹',
    'Turkey': '🇹🇷', 'Iran': '🇮🇷', 'Saudi Arabia': '🇸🇦', 'Qatar': '🇶🇦',
    'Paraguay': '🇵🇾', 'South Africa': '🇿🇦', 'Ghana': '🇬🇭', 'Ivory Coast': '🇨🇮',
    'Tunisia': '🇹🇳', 'Algeria': '🇩🇿', 'Egypt': '🇪🇬', 'Czech Republic': '🇨🇿',
    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'Sweden': '🇸🇪', 'Bosnia and Herzegovina': '🇧🇦',
    'Haiti': '🇭🇹', 'Curaçao': '🇨🇼', 'New Zealand': '🇳🇿', 'Cape Verde': '🇨🇻',
    'DR Congo': '🇨🇩', 'Uzbekistan': '🇺🇿', 'Jordan': '🇯🇴', 'Iraq': '🇮🇶',
    'Panama': '🇵🇦',
}

# ── module-level state (reset on each load_and_train) ────────────────────────
elo          = defaultdict(lambda: 1500.0)
recent       = defaultdict(lambda: deque(maxlen=10))
h2h          = defaultdict(lambda: [0, 0, 0])
wc_matches   = defaultdict(int)
elo_history  = defaultdict(lambda: deque(maxlen=6))   # elo BEFORE each match
streak       = defaultdict(int)                        # +N = win streak, -N = loss streak
all_matches  = []

final_model       = None
train_df          = None
future_df         = None
played_wc2026     = None
last_updated      = None
cached_champ_odds = None
_prob_cache       = {}     # precomputed matchup probabilities for Monte Carlo
_wc_percentiles   = {}     # percentile-based att/def scores for all 48 WC teams


# ── helpers ──────────────────────────────────────────────────────────────────

def k_factor(tourn):
    t = tourn.lower()
    if t == "fifa world cup": return 60
    if "qualification" in t: return 40
    if any(s in t for s in ["copa", "euro", "african", "africa cup",
                             "afc asian", "gold cup", "confederations",
                             "nations league"]): return 50
    if t == "friendly": return 20
    return 30

def mov_mult(gd):
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return (11 + gd) / 8.0

def pair_key(a, b):
    return (a, b) if a < b else (b, a)

def features_for(home, away, neutral, tournament):
    e_h, e_a = elo[home], elo[away]
    is_home  = 0.0 if neutral else 1.0

    rh, ra = recent[home], recent[away]
    def agg(dq):
        if not dq: return 1.0, 1.2, 1.2
        arr = np.array(dq)
        return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
    pts_h, gf_h, ga_h = agg(rh)
    pts_a, gf_a, ga_a = agg(ra)

    k = pair_key(home, away)
    wA, d, wB = h2h[k]
    n = wA + d + wB
    wh = wA if k[0] == home else wB
    wa = wB if k[0] == home else wA
    h2h_score = (wh - wa) / n if n else 0.0

    # Elo momentum: how much has each team's elo changed over the last 6 games
    hist_h = list(elo_history[home])
    hist_a = list(elo_history[away])
    mom_h  = (e_h - hist_h[0]) if len(hist_h) >= 3 else 0.0
    mom_a  = (e_a - hist_a[0]) if len(hist_a) >= 3 else 0.0

    return dict(
        elo_diff          = e_h - e_a,
        is_home           = is_home,
        form_diff         = pts_h - pts_a,
        attack_diff       = gf_h - gf_a,
        defense_diff      = ga_a - ga_h,
        h2h_score         = h2h_score,
        h2h_n             = np.log1p(n),
        wc_exp_diff       = np.log1p(wc_matches[home]) - np.log1p(wc_matches[away]),
        elo_momentum_diff = mom_h - mom_a,
        streak_diff       = float(streak[home] - streak[away]),
    )

def update_state(home, away, hs, as_, neutral, tournament):
    # record elo snapshot BEFORE this match (used for momentum)
    elo_history[home].append(elo[home])
    elo_history[away].append(elo[away])

    d        = (elo[home] + (0 if neutral else HFA_ELO)) - elo[away]
    exp_home = 1.0 / (1.0 + 10 ** (-d / 400.0))
    res      = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
    delta    = k_factor(tournament) * mov_mult(abs(hs - as_)) * (res - exp_home)
    elo[home] += delta
    elo[away] -= delta

    ph = 3 if hs > as_ else (1 if hs == as_ else 0)
    pa = 3 if as_ > hs else (1 if hs == as_ else 0)
    recent[home].append((ph, hs, as_))
    recent[away].append((pa, as_, hs))

    k = pair_key(home, away)
    if hs > as_:   h2h[k][0 if k[0] == home else 2] += 1
    elif as_ > hs: h2h[k][0 if k[0] == away else 2] += 1
    else:          h2h[k][1] += 1

    if tournament == "FIFA World Cup":
        wc_matches[home] += 1
        wc_matches[away] += 1

    # update streaks
    if hs > as_:
        streak[home] = streak[home] + 1 if streak[home] > 0 else 1
        streak[away] = streak[away] - 1 if streak[away] < 0 else -1
    elif as_ > hs:
        streak[away] = streak[away] + 1 if streak[away] > 0 else 1
        streak[home] = streak[home] - 1 if streak[home] < 0 else -1
    else:
        streak[home] = 0
        streak[away] = 0

    all_matches.append({
        "home": home, "away": away,
        "hs": hs, "as_": as_,
        "date": None, "tournament": tournament,
    })

def fit(data):
    lr  = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.1, random_state=RNG))
    hgb = HistGradientBoostingClassifier(
            max_iter=400, max_depth=3, learning_rate=0.04,
            min_samples_leaf=8, random_state=RNG)
    rf  = RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=RNG)
    vc  = VotingClassifier(
            estimators=[('lr', lr), ('hgb', hgb), ('rf', rf)],
            voting='soft', weights=[1, 2, 1.5])
    vc.fit(data[FEATS], data["outcome"])
    return vc


# ── load & train ─────────────────────────────────────────────────────────────

def load_and_train(csv_path):
    global elo, recent, h2h, wc_matches, elo_history, streak, all_matches
    global final_model, train_df, future_df, played_wc2026
    global last_updated, cached_champ_odds, _prob_cache, _wc_percentiles

    elo         = defaultdict(lambda: 1500.0)
    recent      = defaultdict(lambda: deque(maxlen=10))
    h2h         = defaultdict(lambda: [0, 0, 0])
    wc_matches  = defaultdict(int)
    elo_history = defaultdict(lambda: deque(maxlen=6))
    streak      = defaultdict(int)
    all_matches = []

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    future_df = df[df.home_score.isna()].copy()
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df[["home_score", "away_score"]] = df[["home_score", "away_score"]].astype(int)

    rows = []
    for r in df.itertuples(index=False):
        if r.tournament == "FIFA World Cup" and r.date.year >= 1974:
            f = features_for(r.home_team, r.away_team, r.neutral, r.tournament)
            f.update({
                "date": r.date, "year": r.date.year,
                "home_team": r.home_team, "away_team": r.away_team,
                "outcome": 2 if r.home_score > r.away_score
                           else (1 if r.home_score == r.away_score else 0),
            })
            rows.append(f)
        update_state(r.home_team, r.away_team, r.home_score, r.away_score,
                     r.neutral, r.tournament)

    train_df      = pd.DataFrame(rows)
    final_model   = fit(train_df)
    played_wc2026 = train_df[train_df.year == 2026]
    last_updated  = pd.Timestamp.now()

    _prob_cache       = _build_prob_cache()
    _wc_percentiles   = _build_wc_percentiles()
    cached_champ_odds = monte_carlo_champion(n_sims=2000)


# ── Monte Carlo championship simulation ──────────────────────────────────────

def _build_prob_cache():
    all_teams = sorted({t for teams in WC2026_GROUPS.values() for t in teams})
    cache = {}
    for h in all_teams:
        for a in all_teams:
            if h == a:
                continue
            X = pd.DataFrame([features_for(h, a, True, "FIFA World Cup")])[FEATS]
            cache[(h, a)] = final_model.predict_proba(X)[0]  # (p_loss, p_draw, p_win)
    return cache

def _build_wc_percentiles():
    """
    Rank all 48 WC 2026 teams against each other for attacking threat and
    defensive solidity, so scores reflect real quality gaps (France >> Austria).
    Composite: 50% Elo percentile + 50% recent-stats percentile.
    """
    all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
    raw = {}
    for team in all_teams:
        rq = list(recent[team])
        if rq:
            arr = np.array(rq)
            gf  = float(arr[:, 1].mean())
            ga  = float(arr[:, 2].mean())
        else:
            gf, ga = 1.0, 1.5
        raw[team] = {"elo": elo[team], "gf": gf, "ga": ga}

    elo_vals = [raw[t]["elo"] for t in all_teams]
    gf_vals  = [raw[t]["gf"]  for t in all_teams]
    ga_vals  = [raw[t]["ga"]  for t in all_teams]
    n        = len(all_teams)

    def pct(val, vals, higher_is_better=True):
        rank = sorted(vals).index(
            min(vals, key=lambda v: abs(v - val))
        )
        p = rank / (n - 1)
        return p if higher_is_better else 1 - p

    result = {}
    for team in all_teams:
        elo_p = pct(raw[team]["elo"], elo_vals, True)
        gf_p  = pct(raw[team]["gf"],  gf_vals,  True)
        ga_p  = pct(raw[team]["ga"],  ga_vals,  False)  # lower ga = better defense
        result[team] = {
            "attacking_threat":   round((0.5 * elo_p + 0.5 * gf_p)  * 99) + 1,
            "defensive_solidity": round((0.5 * elo_p + 0.5 * ga_p)  * 99) + 1,
        }
    return result


def monte_carlo_champion(n_sims=2000):
    played = {}
    if played_wc2026 is not None:
        for _, row in played_wc2026.iterrows():
            played[(row.home_team, row.away_team)] = row.outcome

    wins = defaultdict(int)

    for _ in range(n_sims):
        # ── group stage ──
        grp_pts   = {}
        grp_order = {}
        for grp, teams in WC2026_GROUPS.items():
            pts = {t: 0 for t in teams}
            for i, t1 in enumerate(teams):
                for t2 in teams[i+1:]:
                    if (t1, t2) in played:
                        o = played[(t1, t2)]
                    elif (t2, t1) in played:
                        o = {2: 0, 1: 1, 0: 2}[played[(t2, t1)]]
                    else:
                        probs = _prob_cache[(t1, t2)]
                        o = int(np.random.choice(3, p=probs))
                    if o == 2:   pts[t1] += 3
                    elif o == 1: pts[t1] += 1; pts[t2] += 1
                    else:        pts[t2] += 3
            grp_pts.update(pts)
            grp_order[grp] = sorted(teams, key=lambda t: (pts[t], elo[t]), reverse=True)

        q1    = [grp_order[g][0] for g in WC2026_GROUPS]
        q2    = [grp_order[g][1] for g in WC2026_GROUPS]
        best3 = sorted(
            [grp_order[g][2] for g in WC2026_GROUPS],
            key=lambda t: (grp_pts[t], elo[t]), reverse=True
        )[:8]
        all_32 = sorted(q1 + q2 + best3, key=lambda t: elo[t], reverse=True)

        # ── knockout rounds ──
        current = all_32
        while len(current) > 1:
            next_round = []
            for i in range(0, len(current), 2):
                h, a  = current[i], current[i+1]
                pl, pd_, pw = _prob_cache[(h, a)]
                ph = pw + pd_ * pw / (pw + pl)
                next_round.append(h if np.random.random() < ph else a)
            current = next_round

        wins[current[0]] += 1

    results = sorted(
        [{"team": t, "prob": round(c / n_sims, 4),
          "flag": FLAGS.get(t, "🏳️"), "elo": round(elo[t])}
         for t, c in wins.items()],
        key=lambda x: x["prob"], reverse=True
    )
    return results


# ── team statistics ───────────────────────────────────────────────────────────

def _elo_sorted():
    return sorted(elo.keys(), key=lambda t: elo[t], reverse=True)

def _radar_scores(stats):
    e    = min(100, max(0, (stats["elo"] - 1400) / 9))
    att  = min(100, stats["gf_per_game"] / 3.0 * 100)
    def_ = min(100, max(0, 100 - stats["ga_per_game"] / 3.0 * 100))
    tot  = stats["wins"] + stats["draws"] + stats["losses"]
    frm  = (stats["wins"] * 3 + stats["draws"]) / (tot * 3) * 100 if tot else 50
    exp  = min(100, np.log1p(stats["wc_matches"]) / np.log1p(120) * 100)
    return [round(e), round(att), round(def_), round(frm),
            stats["set_piece_score"], round(exp)]

def get_team_stats(team):
    rq = list(recent[team])
    if not rq:
        ranked = _elo_sorted()
        return {
            "elo": round(elo[team]),
            "elo_rank": ranked.index(team) + 1 if team in ranked else "—",
            "form_str": "—", "wins": 0, "draws": 0, "losses": 0,
            "gf_per_game": 0.0, "ga_per_game": 0.0, "clean_sheets": 0,
            "wc_matches": int(wc_matches[team]),
            "set_piece_score": 50, "defensive_solidity": 50,
            "attacking_threat": 50, "passing_style": "—",
            "streak": int(streak[team]),
            "elo_momentum": 0,
        }

    arr  = np.array(rq)
    wins   = int((arr[:, 0] == 3).sum())
    draws  = int((arr[:, 0] == 1).sum())
    losses = int((arr[:, 0] == 0).sum())
    gf     = float(arr[:, 1].mean())
    ga     = float(arr[:, 2].mean())
    cs     = int((arr[:, 2] == 0).sum())

    form_str = "".join(
        "W" if p == 3 else "D" if p == 1 else "L"
        for p in arr[-5:, 0]
    )

    ranked   = _elo_sorted()
    elo_rank = ranked.index(team) + 1 if team in ranked else "—"

    wins_by1 = sum(
        1 for m in all_matches
        if (m["home"] == team and m["hs"] - m["as_"] == 1) or
           (m["away"] == team and m["as_"] - m["hs"] == 1)
    )
    total_wins_all = wins_by1 + sum(
        1 for m in all_matches
        if (m["home"] == team and m["hs"] - m["as_"] > 1) or
           (m["away"] == team and m["as_"] - m["hs"] > 1)
    )
    sp_score     = round(min(100, (wins_by1 / total_wins_all * 130) if total_wins_all else 50))
    pct          = _wc_percentiles.get(team, {})
    def_solidity = pct.get("defensive_solidity", round(min(100, max(1, (1.2 - ga) / 1.2 * 99 + 1))))
    att_threat   = pct.get("attacking_threat",   round(min(100, max(1, gf / 3.0 * 99 + 1))))

    if gf >= 1.8 and ga <= 0.8:
        passing_style = "Possession"
    elif gf >= 1.6:
        passing_style = "Direct / High-Press"
    elif ga <= 0.7:
        passing_style = "Defensive / Counter"
    else:
        passing_style = "Balanced"

    hist = list(elo_history[team])
    elo_mom = round(elo[team] - hist[0]) if len(hist) >= 3 else 0

    stats = {
        "elo": round(elo[team]), "elo_rank": elo_rank,
        "form_str": form_str,
        "wins": wins, "draws": draws, "losses": losses,
        "gf_per_game": round(gf, 2), "ga_per_game": round(ga, 2),
        "clean_sheets": cs,
        "wc_matches": int(wc_matches[team]),
        "set_piece_score": sp_score,
        "defensive_solidity": def_solidity,
        "attacking_threat": att_threat,
        "passing_style": passing_style,
        "streak": int(streak[team]),
        "elo_momentum": elo_mom,
    }
    stats["radar"] = _radar_scores(stats)
    return stats

def get_h2h(home, away):
    k  = pair_key(home, away)
    wA, d, wB = h2h[k]
    wh = wA if k[0] == home else wB
    wa = wB if k[0] == home else wA
    return {"wins_home": wh, "draws": d, "wins_away": wa, "total": wh + d + wa}


# ── prediction & explanation ──────────────────────────────────────────────────

def predict_match(home, away, neutral=True):
    f = features_for(home, away, neutral, "FIFA World Cup")
    X = pd.DataFrame([f])[FEATS]
    p_loss, p_draw, p_win = final_model.predict_proba(X)[0]

    ranked    = _elo_sorted()
    elo_rank_h = ranked.index(home) + 1 if home in ranked else "—"
    elo_rank_a = ranked.index(away) + 1 if away in ranked else "—"
    elo_h, elo_a = elo[home], elo[away]
    elo_diff = elo_h - elo_a

    reasons = []

    # 1 — global ranking
    if abs(elo_diff) > 200:
        stronger = home if elo_diff > 0 else away
        reasons.append(f"<b>{stronger}</b> holds a dominant global ranking advantage "
                       f"(#{elo_rank_h} vs #{elo_rank_a}, Elo gap: {abs(elo_diff):.0f} pts). "
                       "Large Elo gaps are the strongest single predictor of match results.")
    elif abs(elo_diff) > 80:
        stronger = home if elo_diff > 0 else away
        reasons.append(f"<b>{stronger}</b> holds a meaningful ranking edge "
                       f"(#{elo_rank_h} vs #{elo_rank_a}, gap: {abs(elo_diff):.0f} pts).")
    else:
        reasons.append(f"Teams are closely matched on global ranking "
                       f"(#{elo_rank_h} vs #{elo_rank_a}, gap: {abs(elo_diff):.0f} pts).")

    # 2 — Elo momentum (new feature)
    hist_h = list(elo_history[home])
    hist_a = list(elo_history[away])
    if len(hist_h) >= 3 and len(hist_a) >= 3:
        mom_h = round(elo_h - hist_h[0])
        mom_a = round(elo_a - hist_a[0])
        if mom_h > 30 and mom_h > mom_a + 20:
            reasons.append(f"<b>{home}</b> is on a strong upward Elo trajectory "
                           f"(+{mom_h} pts over last 6 matches) vs "
                           f"{away} ({'+' if mom_a>=0 else ''}{mom_a} pts) — "
                           "momentum is a key differentiator in knockout football.")
        elif mom_a > 30 and mom_a > mom_h + 20:
            reasons.append(f"<b>{away}</b> is on a strong upward trajectory "
                           f"(+{mom_a} pts) vs {home} ({'+' if mom_h>=0 else ''}{mom_h} pts).")
        elif mom_h < -20 and mom_h < mom_a - 20:
            reasons.append(f"<b>{home}</b> is in declining form "
                           f"({mom_h} Elo pts over last 6 games), a warning sign.")
        elif mom_a < -20 and mom_a < mom_h - 20:
            reasons.append(f"<b>{away}</b> is in declining form ({mom_a} Elo pts).")

    # 3 — win streak (new feature)
    sk_h, sk_a = streak[home], streak[away]
    if sk_h >= 5:
        reasons.append(f"<b>{home}</b> is on a {sk_h}-game winning streak — "
                       "teams in long winning runs carry enormous psychological momentum.")
    elif sk_a >= 5:
        reasons.append(f"<b>{away}</b> is on a {sk_a}-game winning streak.")
    elif sk_h >= 3:
        reasons.append(f"<b>{home}</b> has won {sk_h} straight games.")
    elif sk_a >= 3:
        reasons.append(f"<b>{away}</b> has won {sk_a} straight games.")

    # 4 — recent form
    rh, ra = list(recent[home]), list(recent[away])
    if rh and ra:
        arr_h, arr_a = np.array(rh), np.array(ra)
        form_h, form_a = arr_h[:, 0].mean(), arr_a[:, 0].mean()
        gf_h, ga_h = float(arr_h[:, 1].mean()), float(arr_h[:, 2].mean())
        gf_a, ga_a = float(arr_a[:, 1].mean()), float(arr_a[:, 2].mean())
        fstr_h = "".join("W" if p==3 else "D" if p==1 else "L" for p in arr_h[-5:,0])
        fstr_a = "".join("W" if p==3 else "D" if p==1 else "L" for p in arr_a[-5:,0])

        if form_h > form_a + 0.5:
            reasons.append(f"<b>{home}</b> is in significantly better recent form "
                           f"({fstr_h} vs {fstr_a}).")
        elif form_a > form_h + 0.5:
            reasons.append(f"<b>{away}</b> is in significantly better recent form "
                           f"({fstr_a} vs {fstr_h}).")
        else:
            reasons.append(f"Recent form is closely matched ({fstr_h} vs {fstr_a}).")

        # 5 — attack
        if gf_h > gf_a + 0.6:
            reasons.append(f"<b>{home}</b> has a sharper attack "
                           f"({gf_h:.1f} vs {gf_a:.1f} goals/game last 10).")
        elif gf_a > gf_h + 0.6:
            reasons.append(f"<b>{away}</b> has a sharper attack "
                           f"({gf_a:.1f} vs {gf_h:.1f} goals/game last 10).")

        # 6 — defense
        if ga_h < ga_a - 0.5:
            reasons.append(f"<b>{home}</b> has the tighter defense "
                           f"({ga_h:.1f} vs {ga_a:.1f} goals conceded/game). "
                           "Defensive organization is decisive in knockout rounds.")
        elif ga_a < ga_h - 0.5:
            reasons.append(f"<b>{away}</b> has the tighter defense "
                           f"({ga_a:.1f} vs {ga_h:.1f} conceded/game).")

    # 7 — set-piece threat
    hs = get_team_stats(home)
    as_ = get_team_stats(away)
    sp_h, sp_a = hs["set_piece_score"], as_["set_piece_score"]
    if sp_h > sp_a + 20:
        reasons.append(f"<b>{home}</b> has a higher set-piece threat index ({sp_h} vs {sp_a}/100) — "
                       "they win a disproportionate number of tight games via dead-ball situations.")
    elif sp_a > sp_h + 20:
        reasons.append(f"<b>{away}</b> has a higher set-piece threat index ({sp_a} vs {sp_h}/100).")

    # 8 — head-to-head
    h2h_s = get_h2h(home, away)
    tot   = h2h_s["total"]
    if tot >= 5:
        wh, wa, dd = h2h_s["wins_home"], h2h_s["wins_away"], h2h_s["draws"]
        if wh > wa + 1:
            reasons.append(f"<b>{home}</b> dominates the all-time H2H "
                           f"({wh}W–{dd}D–{wa}L across {tot} meetings).")
        elif wa > wh + 1:
            reasons.append(f"<b>{away}</b> dominates the all-time H2H "
                           f"({wa}W–{dd}D–{wh}L across {tot} meetings).")
        else:
            reasons.append(f"H2H is evenly balanced ({wh}W–{dd}D–{wa}L across {tot} meetings).")
    elif tot > 0:
        reasons.append(f"Only {tot} prior meeting(s) between these sides.")
    else:
        reasons.append("These teams have never faced each other before.")

    # 9 — WC experience
    wc_h, wc_a = wc_matches[home], wc_matches[away]
    if wc_h > wc_a + 20:
        reasons.append(f"<b>{home}</b> has far more World Cup experience "
                       f"({wc_h} vs {wc_a} career WC appearances).")
    elif wc_a > wc_h + 20:
        reasons.append(f"<b>{away}</b> has far more World Cup experience "
                       f"({wc_a} vs {wc_h} career WC appearances).")

    # 10 — playing style
    reasons.append(f"Playing styles: <b>{home}</b> → <i>{hs['passing_style']}</i> · "
                   f"<b>{away}</b> → <i>{as_['passing_style']}</i>.")

    pick      = home if p_win >= p_loss else away
    conf_pct  = max(float(p_win), float(p_loss))
    if conf_pct >= 0.65:
        confidence_label, confidence_cls = "High", "conf-high"
    elif conf_pct >= 0.50:
        confidence_label, confidence_cls = "Moderate", "conf-mid"
    else:
        confidence_label, confidence_cls = "Low", "conf-low"

    home_stats = get_team_stats(home)
    away_stats = get_team_stats(away)

    return {
        "home": home, "away": away,
        "p_home_win":  round(float(p_win), 3),
        "p_draw":      round(float(p_draw), 3),
        "p_away_win":  round(float(p_loss), 3),
        "pick":            pick,
        "confidence_label": confidence_label,
        "confidence_cls":   confidence_cls,
        "reasons":     reasons,
        "home_stats":  home_stats,
        "away_stats":  away_stats,
        "h2h":         h2h_s,
        "home_flag":   FLAGS.get(home, "🏳️"),
        "away_flag":   FLAGS.get(away, "🏳️"),
        "radar_home":  home_stats.get("radar", [50]*6),
        "radar_away":  away_stats.get("radar", [50]*6),
    }


def get_upcoming_games(days_ahead=7):
    if future_df is None:
        return []
    today  = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)
    mask   = (
        (future_df.tournament == "FIFA World Cup") &
        (future_df.date >= today) &
        (future_df.date <= cutoff)
    )
    results = []
    for r in future_df[mask].sort_values("date").itertuples(index=False):
        pred = predict_match(r.home_team, r.away_team, r.neutral)
        pred["date_str"] = r.date.strftime("%b %d, %Y")
        pred["city"]     = r.city
        results.append(pred)
    return results


# ── bracket simulation (deterministic) ───────────────────────────────────────

def simulate_bracket():
    played = {}
    if played_wc2026 is not None:
        for _, row in played_wc2026.iterrows():
            played[(row.home_team, row.away_team)] = row.outcome

    def grp_outcome(home, away):
        if (home, away) in played:
            return played[(home, away)]
        if (away, home) in played:
            return {2: 0, 1: 1, 0: 2}[played[(away, home)]]
        X = pd.DataFrame([features_for(home, away, True, "FIFA World Cup")])[FEATS]
        return int(np.argmax(final_model.predict_proba(X)[0]))

    def ko_winner(home, away):
        X  = pd.DataFrame([features_for(home, away, True, "FIFA World Cup")])[FEATS]
        pl, pd_, pw = final_model.predict_proba(X)[0]
        ph = pw + pd_ * pw / (pw + pl)
        return (home, round(ph, 3)) if ph >= 0.5 else (away, round(1 - ph, 3))

    grp_pts, grp_order = {}, {}
    for grp, teams in WC2026_GROUPS.items():
        pts = {t: 0 for t in teams}
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                o = grp_outcome(t1, t2)
                if o == 2:   pts[t1] += 3
                elif o == 1: pts[t1] += 1; pts[t2] += 1
                else:        pts[t2] += 3
        grp_pts.update(pts)
        grp_order[grp] = sorted(teams, key=lambda t: (pts[t], elo[t]), reverse=True)

    q1    = [grp_order[g][0] for g in WC2026_GROUPS]
    q2    = [grp_order[g][1] for g in WC2026_GROUPS]
    best3 = sorted([grp_order[g][2] for g in WC2026_GROUPS],
                   key=lambda t: (grp_pts[t], elo[t]), reverse=True)[:8]
    all_32 = sorted(q1 + q2 + best3, key=lambda t: elo[t], reverse=True)

    groups_out = {
        grp: [{"team": t, "pts": grp_pts[t], "flag": FLAGS.get(t, "🏳️"),
               "advanced": i < 2, "elo": round(elo[t])}
              for i, t in enumerate(teams)]
        for grp, teams in grp_order.items()
    }

    def sim_round(pairs):
        results, winners = [], []
        for h, a in pairs:
            w, prob = ko_winner(h, a)
            results.append({"home": h, "away": a, "winner": w,
                            "loser": a if w == h else h, "prob": prob,
                            "home_flag": FLAGS.get(h, "🏳️"),
                            "away_flag": FLAGS.get(a, "🏳️")})
            winners.append(w)
        return results, winners

    r32_res, r16t = sim_round([(all_32[i], all_32[31-i]) for i in range(16)])
    r16_res, qft  = sim_round(list(zip(r16t[::2], r16t[1::2])))
    qf_res,  sft  = sim_round(list(zip(qft[::2],  qft[1::2])))
    sf_res,  fint = sim_round(list(zip(sft[::2],  sft[1::2])))
    fin_res, chmp = sim_round([(fint[0], fint[1])])

    return {
        "groups": groups_out,
        "rounds": [
            {"name": "Round of 32",    "matches": r32_res},
            {"name": "Round of 16",    "matches": r16_res},
            {"name": "Quarter-Finals", "matches": qf_res},
            {"name": "Semi-Finals",    "matches": sf_res},
            {"name": "Final",          "matches": fin_res},
        ],
        "champion":      chmp[0],
        "champion_flag": FLAGS.get(chmp[0], "🏳️"),
    }
