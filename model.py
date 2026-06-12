import os, ssl, urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
HFA_ELO = 100
RNG = 42
FEATS = ["elo_diff", "is_home", "defense_diff", "wc_exp_diff"]

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
    'Panama': '🇵🇦', 'Peru': '🇵🇪', 'Chile': '🇨🇱',
}

# Module-level mutable state — reset on each load_and_train call
elo          = defaultdict(lambda: 1500.0)
recent       = defaultdict(lambda: deque(maxlen=10))
h2h          = defaultdict(lambda: [0, 0, 0])
wc_matches   = defaultdict(int)
all_matches  = []          # every played match row for style analysis
final_model  = None
train_df     = None
future_df    = None
played_wc2026 = None
last_updated = None


# ── helpers ─────────────────────────────────────────────────────────────────

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
    is_home = 0.0 if neutral else 1.0
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
    wins_home = wA if k[0] == home else wB
    wins_away = wB if k[0] == home else wA
    h2h_score = (wins_home - wins_away) / n if n else 0.0

    return dict(
        elo_diff=(e_h - e_a),
        is_home=is_home,
        form_diff=pts_h - pts_a,
        attack_diff=gf_h - gf_a,
        defense_diff=ga_a - ga_h,
        h2h_score=h2h_score,
        h2h_n=np.log1p(n),
        wc_exp_diff=np.log1p(wc_matches[home]) - np.log1p(wc_matches[away]),
    )

def update_state(home, away, hs, as_, neutral, tournament):
    d = (elo[home] + (0 if neutral else HFA_ELO)) - elo[away]
    exp_home = 1.0 / (1.0 + 10 ** (-d / 400.0))
    res = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
    delta = k_factor(tournament) * mov_mult(abs(hs - as_)) * (res - exp_home)
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

def fit(data):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, C=0.05, random_state=RNG))
    m.fit(data[FEATS], data["outcome"])
    return m


# ── load & train ─────────────────────────────────────────────────────────────

def load_and_train(csv_path):
    global elo, recent, h2h, wc_matches, all_matches
    global final_model, train_df, future_df, played_wc2026, last_updated

    elo         = defaultdict(lambda: 1500.0)
    recent      = defaultdict(lambda: deque(maxlen=10))
    h2h         = defaultdict(lambda: [0, 0, 0])
    wc_matches  = defaultdict(int)
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
        all_matches.append({
            "home": r.home_team, "away": r.away_team,
            "hs": r.home_score, "as_": r.away_score,
            "date": r.date, "tournament": r.tournament,
        })

    train_df      = pd.DataFrame(rows)
    final_model   = fit(train_df)
    played_wc2026 = train_df[train_df.year == 2026]
    last_updated  = pd.Timestamp.now()


# ── team statistics ──────────────────────────────────────────────────────────

def _elo_sorted():
    return sorted(elo.keys(), key=lambda t: elo[t], reverse=True)

def get_team_stats(team):
    rq = list(recent[team])
    if not rq:
        return {
            "elo": round(elo[team]), "elo_rank": "—",
            "form_str": "N/A", "wins": 0, "draws": 0, "losses": 0,
            "gf_per_game": 0.0, "ga_per_game": 0.0,
            "clean_sheets": 0, "wc_matches": int(wc_matches[team]),
            "set_piece_score": 50, "defensive_solidity": 50,
            "attacking_threat": 50, "passing_style": "—",
        }
    arr = np.array(rq)
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

    # rank among all teams we have elo for
    ranked = _elo_sorted()
    elo_rank = ranked.index(team) + 1 if team in ranked else "—"

    # set-piece proxy: % of wins that were single-goal margins
    # (dead balls very often decide 1-0 / 2-1 games)
    wins_by_1 = sum(
        1 for m in all_matches
        if m["home"] == team and m["hs"] - m["as_"] == 1
        or m["away"] == team and m["as_"] - m["hs"] == 1
    )
    total_wins = wins_by_1 + sum(
        1 for m in all_matches
        if m["home"] == team and m["hs"] - m["as_"] > 1
        or m["away"] == team and m["as_"] - m["hs"] > 1
    )
    sp_score = round(min(100, (wins_by_1 / total_wins * 130) if total_wins else 50))

    # defensive solidity: invert goals-conceded relative to league avg (1.2)
    def_solidity = round(min(100, max(0, (1.2 - ga) / 1.2 * 100 + 50)))

    # attacking threat: goals per game relative to avg (1.2)
    att_threat = round(min(100, max(0, gf / 1.2 * 55)))

    # passing style proxy: high attack & high defense together = possession team
    if gf >= 1.8 and ga <= 0.8:
        passing_style = "Possession"
    elif gf >= 1.6:
        passing_style = "Direct / High-Press"
    elif ga <= 0.7:
        passing_style = "Defensive / Counter"
    else:
        passing_style = "Balanced"

    return {
        "elo": round(elo[team]),
        "elo_rank": elo_rank,
        "form_str": form_str,
        "wins": wins, "draws": draws, "losses": losses,
        "gf_per_game": round(gf, 2),
        "ga_per_game": round(ga, 2),
        "clean_sheets": cs,
        "wc_matches": int(wc_matches[team]),
        "set_piece_score": sp_score,
        "defensive_solidity": def_solidity,
        "attacking_threat": att_threat,
        "passing_style": passing_style,
    }

def get_h2h(home, away):
    k = pair_key(home, away)
    wA, d, wB = h2h[k]
    wh = wA if k[0] == home else wB
    wa = wB if k[0] == home else wA
    return {"wins_home": wh, "draws": d, "wins_away": wa, "total": wh + d + wa}


# ── prediction & explanation ─────────────────────────────────────────────────

def predict_match(home, away, neutral=True):
    f = features_for(home, away, neutral, "FIFA World Cup")
    X = pd.DataFrame([f])[FEATS]
    p_loss, p_draw, p_win = final_model.predict_proba(X)[0]

    ranked = _elo_sorted()
    elo_rank_h = ranked.index(home) + 1 if home in ranked else "—"
    elo_rank_a = ranked.index(away) + 1 if away in ranked else "—"
    elo_h, elo_a = elo[home], elo[away]
    elo_diff = elo_h - elo_a

    reasons = []

    # 1 — global ranking / Elo
    if abs(elo_diff) > 200:
        stronger = home if elo_diff > 0 else away
        reasons.append(f"<b>{stronger}</b> has a dominant global ranking advantage "
                       f"(#{elo_rank_h} vs #{elo_rank_a}, Elo gap: {abs(elo_diff):.0f} pts). "
                       "Large Elo gaps are the single strongest predictor of match outcomes.")
    elif abs(elo_diff) > 80:
        stronger = home if elo_diff > 0 else away
        reasons.append(f"<b>{stronger}</b> holds a meaningful ranking edge "
                       f"(#{elo_rank_h} vs #{elo_rank_a}, gap: {abs(elo_diff):.0f} pts).")
    elif abs(elo_diff) > 25:
        stronger = home if elo_diff > 0 else away
        reasons.append(f"<b>{stronger}</b> edges the global ranking "
                       f"(#{elo_rank_h} vs #{elo_rank_a}).")
    else:
        reasons.append(f"Both teams are essentially equal on global ranking "
                       f"(#{elo_rank_h} vs #{elo_rank_a}) — a genuine 50/50.")

    # 2 — recent form
    rh, ra = list(recent[home]), list(recent[away])
    if rh and ra:
        arr_h, arr_a = np.array(rh), np.array(ra)
        form_h, form_a = arr_h[:, 0].mean(), arr_a[:, 0].mean()
        gf_h, ga_h = arr_h[:, 1].mean(), arr_h[:, 2].mean()
        gf_a, ga_a = arr_a[:, 1].mean(), arr_a[:, 2].mean()
        fstr_h = "".join("W" if p==3 else "D" if p==1 else "L" for p in arr_h[-5:,0])
        fstr_a = "".join("W" if p==3 else "D" if p==1 else "L" for p in arr_a[-5:,0])

        if form_h > form_a + 0.5:
            reasons.append(f"<b>{home}</b> is in superior recent form "
                           f"({fstr_h} vs {fstr_a} in last 5).")
        elif form_a > form_h + 0.5:
            reasons.append(f"<b>{away}</b> is in superior recent form "
                           f"({fstr_a} vs {fstr_h} in last 5).")
        else:
            reasons.append(f"Recent form is closely matched "
                           f"({fstr_h} vs {fstr_a} in last 5).")

        # 3 — attack
        if gf_h > gf_a + 0.6:
            reasons.append(f"<b>{home}</b> carries a stronger attacking threat "
                           f"({gf_h:.1f} vs {gf_a:.1f} goals/game over the last 10 matches).")
        elif gf_a > gf_h + 0.6:
            reasons.append(f"<b>{away}</b> carries a stronger attacking threat "
                           f"({gf_a:.1f} vs {gf_h:.1f} goals/game over the last 10 matches).")

        # 4 — defense / defensive solidity
        if ga_h < ga_a - 0.5:
            reasons.append(f"<b>{home}</b> has the tighter defense "
                           f"({ga_h:.1f} vs {ga_a:.1f} goals conceded/game) — "
                           "defensive stability often proves decisive at the World Cup.")
        elif ga_a < ga_h - 0.5:
            reasons.append(f"<b>{away}</b> has the tighter defense "
                           f"({ga_a:.1f} vs {ga_h:.1f} goals conceded/game).")

    # 5 — set-piece threat (proxy from single-goal-margin wins)
    sp_h = get_team_stats(home)["set_piece_score"]
    sp_a = get_team_stats(away)["set_piece_score"]
    if sp_h > sp_a + 20:
        reasons.append(f"<b>{home}</b> shows a stronger set-piece threat index "
                       f"({sp_h} vs {sp_a}/100) — a high proportion of their wins "
                       "come from single-goal margins, a hallmark of dead-ball efficiency.")
    elif sp_a > sp_h + 20:
        reasons.append(f"<b>{away}</b> shows a stronger set-piece threat index "
                       f"({sp_a} vs {sp_h}/100).")

    # 6 — head-to-head
    h2h_s = get_h2h(home, away)
    total = h2h_s["total"]
    if total >= 5:
        wh, wa, dd = h2h_s["wins_home"], h2h_s["wins_away"], h2h_s["draws"]
        if wh > wa + 1:
            reasons.append(f"<b>{home}</b> dominates the head-to-head record "
                           f"({wh}W–{dd}D–{wa}L across {total} meetings).")
        elif wa > wh + 1:
            reasons.append(f"<b>{away}</b> dominates the head-to-head record "
                           f"({wa}W–{dd}D–{wh}L across {total} meetings).")
        else:
            reasons.append(f"Head-to-head history is evenly balanced "
                           f"({wh}W–{dd}D–{wa}L across {total} meetings).")
    elif total > 0:
        reasons.append(f"Only {total} prior meeting(s) between these sides — "
                       "limited historical data.")
    else:
        reasons.append("These teams have never met before — no head-to-head data available.")

    # 7 — World Cup experience
    wc_h, wc_a = wc_matches[home], wc_matches[away]
    if wc_h > wc_a + 20:
        reasons.append(f"<b>{home}</b> has vastly more World Cup experience "
                       f"({wc_h} vs {wc_a} career WC matches played), "
                       "which matters under tournament pressure.")
    elif wc_a > wc_h + 20:
        reasons.append(f"<b>{away}</b> has vastly more World Cup experience "
                       f"({wc_a} vs {wc_h} career WC matches).")
    elif wc_h > wc_a + 8:
        reasons.append(f"<b>{home}</b> has meaningfully more WC experience "
                       f"({wc_h} vs {wc_a} matches).")
    elif wc_a > wc_h + 8:
        reasons.append(f"<b>{away}</b> has meaningfully more WC experience "
                       f"({wc_a} vs {wc_h} matches).")

    # 8 — passing / style
    st_h = get_team_stats(home)["passing_style"]
    st_a = get_team_stats(away)["passing_style"]
    reasons.append(f"Playing styles: <b>{home}</b> classified as <i>{st_h}</i> · "
                   f"<b>{away}</b> classified as <i>{st_a}</i>.")

    pick = home if p_win >= p_loss else away
    conf_pct = max(p_win, p_loss)
    if conf_pct >= 0.65:
        confidence_label = "High"
        confidence_cls   = "conf-high"
    elif conf_pct >= 0.50:
        confidence_label = "Moderate"
        confidence_cls   = "conf-mid"
    else:
        confidence_label = "Low"
        confidence_cls   = "conf-low"

    return {
        "home": home, "away": away,
        "p_home_win": round(float(p_win), 3),
        "p_draw":     round(float(p_draw), 3),
        "p_away_win": round(float(p_loss), 3),
        "pick": pick,
        "confidence_label": confidence_label,
        "confidence_cls":   confidence_cls,
        "reasons": reasons,
        "home_stats": get_team_stats(home),
        "away_stats": get_team_stats(away),
        "h2h": h2h_s,
        "home_flag": FLAGS.get(home, "🏳️"),
        "away_flag": FLAGS.get(away, "🏳️"),
    }


def get_upcoming_games(days_ahead=7):
    if future_df is None:
        return []
    today  = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)
    mask = (
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


# ── bracket simulation ────────────────────────────────────────────────────────

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
        X = pd.DataFrame([features_for(home, away, True, "FIFA World Cup")])[FEATS]
        pl, pd_, pw = final_model.predict_proba(X)[0]
        ph = pw + pd_ * pw / (pw + pl)
        return (home, round(ph, 3)) if ph >= 0.5 else (away, round(1 - ph, 3))

    # group stage
    grp_pts   = {}
    grp_order = {}
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
    best3 = sorted(
        [grp_order[g][2] for g in WC2026_GROUPS],
        key=lambda t: (grp_pts[t], elo[t]), reverse=True
    )[:8]

    all_32      = sorted(q1 + q2 + best3, key=lambda t: elo[t], reverse=True)
    r32_pairs   = [(all_32[i], all_32[31-i]) for i in range(16)]

    groups_out = {}
    for grp, teams in grp_order.items():
        groups_out[grp] = [
            {"team": t, "pts": grp_pts[t],
             "flag": FLAGS.get(t, "🏳️"),
             "advanced": i < 2,
             "elo": round(elo[t])}
            for i, t in enumerate(teams)
        ]

    def sim_round(pairs):
        results, winners = [], []
        for h, a in pairs:
            w, prob = ko_winner(h, a)
            results.append({
                "home": h, "away": a, "winner": w,
                "loser": a if w == h else h,
                "prob": prob,
                "home_flag": FLAGS.get(h, "🏳️"),
                "away_flag": FLAGS.get(a, "🏳️"),
            })
            winners.append(w)
        return results, winners

    r32_res, r16_teams = sim_round(r32_pairs)
    r16_res, qf_teams  = sim_round(list(zip(r16_teams[::2], r16_teams[1::2])))
    qf_res,  sf_teams  = sim_round(list(zip(qf_teams[::2],  qf_teams[1::2])))
    sf_res,  fin_teams = sim_round(list(zip(sf_teams[::2],  sf_teams[1::2])))
    fin_res, champ     = sim_round([(fin_teams[0], fin_teams[1])])

    return {
        "groups": groups_out,
        "rounds": [
            {"name": "Round of 32",     "matches": r32_res},
            {"name": "Round of 16",     "matches": r16_res},
            {"name": "Quarter-Finals",  "matches": qf_res},
            {"name": "Semi-Finals",     "matches": sf_res},
            {"name": "Final",           "matches": fin_res},
        ],
        "champion":      champ[0],
        "champion_flag": FLAGS.get(champ[0], "🏳️"),
    }
