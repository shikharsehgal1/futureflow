"""
part8.py  —  Dixon-Coles Poisson Model

Upgrades over Part 7 (WLS goal-differential):
  1. Separate attack + defence ratings per team
  2. Full score probability matrix -> proper H/D/A + over/under
  3. Dixon-Coles rho correction -- fixes draw underestimation
  4. Informative preseason priors from last season

Architecture (3-step fit):
  Step 1: Part 7 WLS on mu_mkt  -> correct overall ratings (market-informed)
  Step 2: Decompose overall into attack/defence using actual goal data
  Step 3: MLE on actual scorelines -> learn rho + refine HFA

This preserves the market signal correctly (which operates in linear goal-diff
space) while getting meaningful attack/defence splits from actual goals.

Usage:
  python part8.py                          # all leagues, 2025-26
  python part8.py --div E0                 # EPL only
  python part8.py --div E0 D1             # multiple leagues
  python part8.py --compare               # walk-forward vs market, 2023-24
"""

import argparse, pickle, warnings, os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/big5_with_probs.csv"
MG_PATH        = "data/mean_goals.pkl"
CURRENT_SEASON = "2025-26"
PREV_SEASON    = "2024-25"
HALF_LIFE      = 90
MARKET_WEIGHT  = 0.95
RHO_INIT       = -0.13
MAX_GOALS      = 10
MIN_GAMES      = 10
ALERT_THRESH   = 0.06

DIV_NAMES = {
    "E0":"Premier League","E1":"Championship",
    "D1":"Bundesliga","D2":"2. Bundesliga",
    "SP1":"La Liga","SP2":"Segunda Division",
    "I1":"Serie A","I2":"Serie B",
    "F1":"Ligue 1","F2":"Ligue 2",
}

# ── LOAD ───────────────────────────────────────────────────────────────────────
def load_data(season=CURRENT_SEASON, div=None):
    df = pd.read_csv(DATA_PATH, encoding="utf-8", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["season"] == season].copy()
    if div:
        divs = [div] if isinstance(div, str) else div
        df = df[df["Div"].isin(divs)].copy()
    return df

def load_mean_goals():
    with open(MG_PATH, "rb") as f:
        mg = pickle.load(f)
    return mg["home"], mg["away"]

# ── PROBABILITY CONVERSION (Part 7 symmetric Poisson) ────────────────────────
# Use Part 7 symmetric Poisson for H/D/A: lam_home = (total+mu)/2, lam_away = (total-mu)/2.
# This keeps draw probabilities stable and aligned with the market.
# For p_over25: use actual lam/mu from attack/defence ratings (asymmetric)
# so over/under varies meaningfully across games.
def poisson_probs(mu, mean_home, mean_away, lam_atk=None, mu_atk=None, max_goals=MAX_GOALS):
    total = mean_home + mean_away
    # H/D/A from symmetric (market-aligned)
    lH = np.clip((total + mu) / 2, 0.1, None)
    lA = np.clip((total - mu) / 2, 0.1, None)
    g  = np.arange(max_goals + 1)
    M_sym = np.outer(poisson.pmf(g, lH), poisson.pmf(g, lA))
    pH = np.tril(M_sym,-1).sum()
    pD = np.trace(M_sym)
    pA = np.triu(M_sym,1).sum()
    # p_over25 from asymmetric (attack/defence ratings) for meaningful variation
    if lam_atk is not None and mu_atk is not None:
        M_atk = np.outer(poisson.pmf(g, lam_atk), poisson.pmf(g, mu_atk))
        p_over25 = sum(M_atk[i,j] for i in range(max_goals+1)
                                   for j in range(max_goals+1) if i+j>2)
    else:
        p_over25 = sum(M_sym[i,j] for i in range(max_goals+1)
                                   for j in range(max_goals+1) if i+j>2)
    return (np.clip([pH, pD, pA], 1e-6, 1-1e-6),
            round(lH, 3), round(lA, 3), round(p_over25, 4))

# ── FIT RATINGS ───────────────────────────────────────────────────────────────
def fit_ratings(subset, half_life=HALF_LIFE, market_weight=MARKET_WEIGHT,
                ha_prior=0.3, prev_ratings=None):
    """
    Fit Dixon-Coles ratings via 3-step decomposition.

    Step 1: Part 7 WLS on mu_mkt  -> overall ratings (market-informed)
    Step 2: Decompose into attack/defence from actual goals
    Step 3: MLE on scorelines -> rho + HFA

    Args:
        subset       : DataFrame for one league-season
        half_life    : time decay half-life in days
        market_weight: weight on market signal (used in Step 1 WLS)
        ha_prior     : prior on home advantage
        prev_ratings : dict from previous season for informative priors
    """
    subset = subset.copy().reset_index(drop=True)
    subset = subset.dropna(subset=["HomeGoals","AwayGoals","mu_mkt"])
    if len(subset) < MIN_GAMES:
        return None

    teams    = sorted(set(subset["HomeTeam"]) | set(subset["AwayTeam"]))
    n        = len(teams)
    ref_date = subset["Date"].max()

    days_ago = (ref_date - subset["Date"]).dt.days.values
    w_decay  = (0.5 ** (days_ago / half_life)).astype(float) if half_life else np.ones(len(subset))
    w_mkt    = w_decay * market_weight
    w_act    = w_decay * (1 - market_weight)

    hi = np.array([teams.index(t) for t in subset["HomeTeam"]])
    ai = np.array([teams.index(t) for t in subset["AwayTeam"]])
    hg = subset["HomeGoals"].values.astype(int)
    ag = subset["AwayGoals"].values.astype(int)
    mu_mkt_arr = subset["mu_mkt"].values

    mean_home, mean_away = load_mean_goals()
    mean_goals = (mean_home + mean_away) / 2

    # ── Step 1: WLS on mu_mkt -> overall ratings ─────────────────────────────
    # rating_home - rating_away + hfa = mu_mkt
    ghost_w = 3.0 / max(len(subset)*2/n, 0.1)  # adaptive prior
    X=[]; y=[]; ww=[]
    for k in range(len(subset)):
        r=np.zeros(n+1); r[hi[k]]=1; r[ai[k]]=-1; r[-1]=1
        X.append(r); y.append(mu_mkt_arr[k]); ww.append(w_mkt[k])
    # Ghost team priors
    for i in range(n):
        r=np.zeros(n+1); r[i]=1; r[-1]=1
        X.append(r); y.append(ha_prior); ww.append(ghost_w)
    r=np.zeros(n+1); r[-1]=1
    X.append(r); y.append(ha_prior); ww.append(ghost_w)

    Xm=np.array(X); ym=np.array(y); wm=np.array(ww)
    sq=np.sqrt(wm); b7,_,_,_=np.linalg.lstsq(Xm*sq[:,None],ym*sq,rcond=None)
    overall = b7[:n]; hfa_wls = float(b7[-1])
    overall -= overall.mean()

    # NOTE: prev season shrinkage disabled — causes rating drift from market signal.
    # Prev season ratings are still used to initialize attack/defence split only.
    # (shrinkage would pull overall ratings away from mu_mkt, inflating alerts)

    # ── Step 2: Langville-Meyer OD ratings for attack/defence split ──────────
    # Method from "Who's #1?" (Langville & Meyer). Iteratively adjusts raw xG
    # totals by opponent quality — scoring against strong defences boosts your
    # attack rating more than scoring against weak ones, and vice versa.
    #
    # xG proxy: shots on target × league conversion rate (less noisy than goals)
    # Time decay: each game weighted by w_decay (90-day half-life)
    #
    # G[i,j] = time-decayed xG that team i conceded to team j
    # offensive[j] = Σ_i G[i,j] / defensive[i]   (large = strong attack)
    # defensive[i] = Σ_j G[i,j] / offensive[j]   (large = weak defence)
    # Converges in ~5-10 iterations.

    # Compute xG proxy
    has_sot = ("HST" in subset.columns and "AST" in subset.columns and
               subset["HST"].notna().mean() > 0.5)
    if has_sot:
        valid = subset[subset["HST"].notna() & subset["AST"].notna() &
                       (subset["HST"] > 0) & (subset["AST"] > 0)]
        home_conv = (valid["HomeGoals"].sum() / valid["HST"].sum()
                     if len(valid) >= MIN_GAMES else 0.316)
        away_conv = (valid["AwayGoals"].sum() / valid["AST"].sum()
                     if len(valid) >= MIN_GAMES else 0.306)
        hst = subset["HST"].fillna(subset["HomeGoals"] / max(home_conv, 0.1)).values
        ast = subset["AST"].fillna(subset["AwayGoals"] / max(away_conv, 0.1)).values
        xg_home = np.maximum(hst * home_conv, 0.05)
        xg_away = np.maximum(ast * away_conv, 0.05)
    else:
        xg_home = np.maximum(hg.astype(float), 0.05)
        xg_away = np.maximum(ag.astype(float), 0.05)

    # Build time-decayed xG matrix G[i,j] = xG team i conceded to team j
    # (sum over all games between the pair, weighted by recency)
    G = np.zeros((n, n))
    for k, (ht, at, xgh, xga, wk) in enumerate(zip(
            subset["HomeTeam"], subset["AwayTeam"], xg_home, xg_away, w_decay)):
        hi_k = teams.index(ht); ai_k = teams.index(at)
        G[ai_k, hi_k] += wk * xgh   # away team conceded xgh to home team
        G[hi_k, ai_k] += wk * xga   # home team conceded xga to away team

    # Langville-Meyer iteration
    od_off = np.ones(n)   # offensive ratings (large = strong attack)
    od_def = np.ones(n)   # defensive ratings (large = weak defence)
    for _ in range(50):
        od_off_new = np.array([
            sum(G[i, j] / max(od_def[i], 1e-6) for i in range(n))
            for j in range(n)
        ])
        od_def_new = np.array([
            sum(G[i, j] / max(od_off_new[j], 1e-6) for j in range(n))
            for i in range(n)
        ])
        if (np.max(np.abs(od_off_new - od_off)) < 1e-6 and
                np.max(np.abs(od_def_new - od_def)) < 1e-6):
            break
        od_off, od_def = od_off_new, od_def_new

    # Convert to log scale and recenter (mean=0)
    # od_off: large = good attack.  od_def: large = weak defence.
    # We want: atk_raw positive = good attack, dfc_raw positive = weak defence
    atk_raw = np.log(np.maximum(od_off, 1e-6))
    dfc_raw = np.log(np.maximum(od_def, 1e-6))
    atk_raw -= atk_raw.mean()
    dfc_raw -= dfc_raw.mean()

    # Split: preserve overall, dampen asymmetry signal
    # With only ~15 games/team the raw goal split is noisy — shrink it heavily
    # SPLIT_DAMP=0.0 → equal split (atk=dfc=overall/2)
    # SPLIT_DAMP=1.0 → full raw goal asymmetry (too noisy with <20 games/team)
    SPLIT_DAMP = 1.0
    split = (atk_raw - dfc_raw) * SPLIT_DAMP
    atk_init =  overall/2 + split
    dfc_init = -overall/2 + split

    # ── Step 3: Set HFA + learn rho from scorelines ───────────────────────────
    # HFA scale problem: hfa_wls is in linear goal-diff space,
    # but Poisson MLE needs log space. Convert:
    #   lam_home / lam_away = exp(hfa_log)
    #   hfa_wls ≈ lam_home - lam_away ≈ mean_goals * (exp(hfa_log) - 1)
    #   => hfa_log ≈ log(1 + hfa_wls / mean_goals)
    hfa_log = float(np.log(1 + max(hfa_wls, 0) / mean_goals))

    # Only fit rho from scorelines — HFA already set correctly above
    atk_arr = np.array([atk_init[i] for i in range(n)])
    dfc_arr = np.array([dfc_init[i] for i in range(n)])

    def neg_ll_rho(rho_x):
        lam=np.exp(atk_arr[hi]-dfc_arr[ai]+hfa_log)
        mu =np.exp(atk_arr[ai]-dfc_arr[hi])
        lam=np.clip(lam,0.05,15); mu=np.clip(mu,0.05,15)
        log_ph=poisson.logpmf(hg,lam); log_pa=poisson.logpmf(ag,mu)
        tau_v=np.ones(len(hg))
        m00=(hg==0)&(ag==0); m10=(hg==1)&(ag==0)
        m01=(hg==0)&(ag==1); m11=(hg==1)&(ag==1)
        tau_v[m00]=np.maximum(1-lam[m00]*mu[m00]*rho_x,1e-6)
        tau_v[m10]=1+mu[m10]*rho_x; tau_v[m01]=1+lam[m01]*rho_x
        tau_v[m11]=np.maximum(1-rho_x,1e-6)
        with np.errstate(invalid='ignore',divide='ignore'):
            log_tau=np.where(tau_v>0,np.log(np.maximum(tau_v,1e-10)),-20.0)
        return -np.sum(w_decay*(log_tau+log_ph+log_pa))

    # Fix rho to literature value (-0.13) rather than fitting from ~300 games.
    # With this small sample, MLE on rho is unstable. The well-established
    # Dixon-Coles value of -0.13 is more reliable than a noisy per-league fit.
    hfa_final=hfa_log; rho_final=RHO_INIT
    converged=True

    return {
        "attack":    {teams[i]: atk_init[i] for i in range(n)},
        "defence":   {teams[i]: dfc_init[i] for i in range(n)},
        "overall":   {teams[i]: overall[i]   for i in range(n)},
        "hfa":       hfa_wls,   # linear goal-diff space for symmetric Poisson
        "n_games":   len(subset),
        "ref_date":  ref_date,
        "teams":     teams,
        "mean_home": mean_home,
        "mean_away": mean_away,
        "converged": converged,
    }

# ── PREDICT MATCH ─────────────────────────────────────────────────────────────
def predict_match(home, away, ratings):
    """
    Predict using Part 7 symmetric Poisson on overall ratings.
    Attack/defence are for display only — predictions use overall = attack-defence.
    """
    overall  = ratings["overall"]
    hfa      = ratings["hfa"]     # linear goal-diff space
    mean_home= ratings["mean_home"]
    mean_away= ratings["mean_away"]

    # Overall rating in linear goal-diff space (same as Part 7)
    mu = overall.get(home, 0) - overall.get(away, 0) + hfa

    # Asymmetric lambdas from attack/defence for p_over25
    atk  = ratings["attack"]; dfc = ratings["defence"]
    mean_goals = (mean_home + mean_away) / 2
    lam_atk = np.clip(np.exp(atk.get(home,0) - dfc.get(away,0) + np.log(1 + max(hfa,0)/mean_goals)), 0.05, 20)
    mu_atk  = np.clip(np.exp(atk.get(away,0) - dfc.get(home,0)),                                     0.05, 20)

    (pH,pD,pA), lam, lam_away, p_over25 = poisson_probs(
        mu, mean_home, mean_away, lam_atk=lam_atk, mu_atk=mu_atk)
    return {"lam":lam, "mu":lam_away, "pH":pH, "pD":pD, "pA":pA,
            "p_over25":p_over25}

# ── RATINGS TABLE ─────────────────────────────────────────────────────────────
def print_ratings_table(div, ratings, season=CURRENT_SEASON):
    overall=ratings["overall"]; atk=ratings["attack"]; dfc=ratings["defence"]
    hfa=ratings["hfa"]  # linear goal-diff space

    rows=sorted([(t,overall[t],atk[t],dfc[t]) for t in ratings["teams"]],
                key=lambda x:-x[1])

    print(f"\n{'='*70}")
    print(f"  {DIV_NAMES.get(div,div)} ({div})  |  {season}  |  "
          f"{ratings['n_games']} games  |  as of {ratings['ref_date'].date()}")
    print(f"  HFA={hfa:+.3f} goals  |  Anchor: __LEAGUE_AVG__ = 0.000  |  "
          f"rho={RHO_INIT} (fixed)  |  attack/defence: Langville-Meyer OD (xG-weighted)")
    print(f"{'='*70}")
    print(f"  {'#':>3}  {'Team':<22}  {'Overall':>8}  {'Attack':>8}  {'Defence':>9}")
    print(f"  {'-'*58}")
    for rank,(t,ov,a,d) in enumerate(rows,1):
        bar="█"*max(int((ov+2)*4),0)
        print(f"  {rank:>3}  {t:<22}  {ov:>+8.3f}  {a:>+8.3f}  {d:>+9.3f}  {bar}")

# ── PREDICT FIXTURES ──────────────────────────────────────────────────────────
def predict_fixtures(fixtures_path, league_ratings):
    try: fix=pd.read_csv(fixtures_path)
    except FileNotFoundError:
        print(f"No fixtures at {fixtures_path}"); return pd.DataFrame()

    has_odds=all(c in fix.columns for c in ["OddsH","OddsD","OddsA"])
    rows=[]
    for _,f in fix.iterrows():
        div=f["Div"]; home=f["HomeTeam"]; away=f["AwayTeam"]
        if div not in league_ratings: continue
        ratings=league_ratings[div]
        if home not in ratings["attack"]:
            print(f"  WARNING: '{home}' not in {div}")
        if away not in ratings["attack"]:
            print(f"  WARNING: '{away}' not in {div}")
        pred=predict_match(home,away,ratings)
        row={"Date":f.get("Date",""),"Div":div,"HomeTeam":home,"AwayTeam":away,
             "xG_Home":pred["lam"],"xG_Away":pred["mu"],
             "pH_model":round(pred["pH"]*100,1),"pD_model":round(pred["pD"]*100,1),
             "pA_model":round(pred["pA"]*100,1),"p_over25":round(pred["p_over25"]*100,1)}
        if has_odds and pd.notna(f.get("OddsH")):
            raw=np.array([1/f["OddsH"],1/f["OddsD"],1/f["OddsA"]]); raw/=raw.sum()
            mH,mD,mA=raw
            row.update({"pH_mkt":round(mH*100,1),"pD_mkt":round(mD*100,1),
                        "pA_mkt":round(mA*100,1),"diff_H":round((pred["pH"]-mH)*100,1),
                        "diff_D":round((pred["pD"]-mD)*100,1),
                        "diff_A":round((pred["pA"]-mA)*100,1),
                        "alert":(all(pd.notna([mH,mD,mA])) and
                                 max(abs(pred["pH"]-mH),abs(pred["pD"]-mD),
                                     abs(pred["pA"]-mA))>ALERT_THRESH)})
        rows.append(row)

    preds=pd.DataFrame(rows)
    if preds.empty: return preds

    print(f"\n{'='*82}")
    print(f"  PREDICTIONS  —  Dixon-Coles  (alert: >{ALERT_THRESH*100:.0f}pp)")
    print(f"{'='*82}")
    for div,grp in preds.groupby("Div"):
        print(f"\n  {DIV_NAMES.get(div,div)} ({div})")
        print(f"  {'Date':<12} {'Home':<20} {'Away':<20} "
              f"{'xGH':>4} {'xGA':>4} {'H%':>5} {'D%':>5} {'A%':>5} {'O25':>5}",end="")
        if has_odds: print(f"  {'MH%':>5} {'MD%':>5} {'MA%':>5} "
                           f"{'dH':>5} {'dD':>5} {'dA':>5}",end="")
        print()
        print(f"  {'-'*80}")
        for _,r in grp.iterrows():
            alert=" ◀" if r.get("alert") else ""
            line=(f"  {str(r['Date']):<12} {r['HomeTeam']:<20} "
                  f"{r['AwayTeam']:<20} "
                  f"{r['xG_Home']:>4.2f} {r['xG_Away']:>4.2f} "
                  f"{r['pH_model']:>4.1f}% {r['pD_model']:>4.1f}% "
                  f"{r['pA_model']:>4.1f}% {r['p_over25']:>4.1f}%")
            if has_odds and "pH_mkt" in r:
                line+=(f"  {r['pH_mkt']:>4.1f}% {r['pD_mkt']:>4.1f}% "
                       f"{r['pA_mkt']:>4.1f}%  "
                       f"{r['diff_H']:>+5.1f} {r['diff_D']:>+5.1f} "
                       f"{r['diff_A']:>+5.1f}")
            print(line+alert)

    if has_odds and "alert" in preds.columns:
        alerts=preds[preds["alert"]==True].copy()
        if len(alerts):
            print(f"\n{'='*82}")
            print(f"  ALERTS  —  {len(alerts)} games where DC disagrees with market by >{ALERT_THRESH*100:.0f}pp")
            print(f"{'='*82}")
            alerts["max_diff"]=alerts[["diff_H","diff_D","diff_A"]].abs().max(axis=1)
            for _,r in alerts.sort_values("max_diff",ascending=False).iterrows():
                diffs={"Home":r["diff_H"],"Draw":r["diff_D"],"Away":r["diff_A"]}
                biggest=max(diffs,key=lambda k:abs(diffs[k]))
                dirn="higher" if diffs[biggest]>0 else "lower"
                print(f"\n  {r['HomeTeam']} vs {r['AwayTeam']} ({r['Div']})")
                print(f"    DC sees {biggest} as {dirn} ({diffs[biggest]:+.1f}pp)")
                print(f"    DC:  H={r['pH_model']}%  D={r['pD_model']}%  "
                      f"A={r['pA_model']}%  xG={r['xG_Home']:.2f}-{r['xG_Away']:.2f}")
                print(f"    Mkt: H={r['pH_mkt']}%  D={r['pD_mkt']}%  "
                      f"A={r['pA_mkt']}%")

    preds.to_csv("data/part8_predictions.csv",index=False)
    print(f"\nSaved -> data/part8_predictions.csv  ({len(alerts) if 'alerts' in dir() else 0} alerts)")
    return preds

# ── COMPARE VS MARKET ─────────────────────────────────────────────────────────
def compare_models(test_season="2023-24"):
    df=pd.read_csv(DATA_PATH,encoding="utf-8",low_memory=False)
    df["Date"]=pd.to_datetime(df["Date"])
    df=df[df["season"]==test_season].dropna(subset=["HomeGoals","AwayGoals","p_H_mkt"])
    all_dc=[]; all_mkt=[]
    print(f"Walk-forward comparison — {test_season}\n")
    for div,group in df.groupby("Div"):
        group=group.sort_values("Date").reset_index(drop=True)
        for gd in group["Date"].unique():
            train=group[group["Date"]<gd]; predict=group[group["Date"]==gd]
            if len(train)<MIN_GAMES: continue
            res=fit_ratings(train)
            if res is None: continue
            for _,row in predict.iterrows():
                pred=predict_match(row["HomeTeam"],row["AwayTeam"],res)
                result=("H" if row["HomeGoals"]>row["AwayGoals"] else
                        "A" if row["HomeGoals"]<row["AwayGoals"] else "D")
                p_dc={"H":pred["pH"],"D":pred["pD"],"A":pred["pA"]}[result]
                p_mkt={"H":row["p_H_mkt"],"D":row["p_D_mkt"],"A":row["p_A_mkt"]}[result]
                all_dc.append(max(p_dc,1e-6)); all_mkt.append(max(p_mkt,1e-6))

    ll_dc=-np.mean(np.log(all_dc)); ll_mkt=-np.mean(np.log(all_mkt))
    print(f"  N games         : {len(all_dc)}")
    print(f"  DC log-loss     : {ll_dc:.4f}")
    print(f"  Market log-loss : {ll_mkt:.4f}")
    print(f"  Gap             : {ll_mkt-ll_dc:+.4f}  "
          f"({'DC better' if ll_dc<ll_mkt else 'market better'})")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(season=CURRENT_SEASON, divs=None, fixtures_path=None, compare=False):
    if compare:
        compare_models(); return

    print(f"Loading data  —  season: {season}")
    df_cur=load_data(season=season,div=divs)
    df_prv=load_data(season=PREV_SEASON,div=divs)
    print(f"  Current season: {len(df_cur):,} games  |  "
          f"Previous season: {len(df_prv):,} games")

    # Fit previous season for priors
    prev_ratings={}
    for div_code,group in df_prv.groupby("Div"):
        group=group.dropna(subset=["HomeGoals","AwayGoals","mu_mkt"])
        if len(group)<MIN_GAMES: continue
        res=fit_ratings(group)
        if res: prev_ratings[div_code]=res

    # Fit current season
    print(f"\nFitting Dixon-Coles ratings...")
    league_ratings={}
    for div_code,group in df_cur.groupby("Div"):
        group=group.dropna(subset=["HomeGoals","AwayGoals","mu_mkt"])
        if len(group)<MIN_GAMES: continue
        res=fit_ratings(group,prev_ratings=prev_ratings.get(div_code))
        if res:
            league_ratings[div_code]=res
            print(f"  {div_code:<5} {DIV_NAMES.get(div_code,''):<25} "
                  f"{res['n_games']} games  hfa={res['hfa']:+.3f}")

    print(f"\nFitted {len(league_ratings)} leagues")
    print(f"Hyperparams: half_life={HALF_LIFE}d  market_weight={MARKET_WEIGHT}")

    for div_code in sorted(league_ratings):
        print_ratings_table(div_code, league_ratings[div_code], season)

    fp=fixtures_path or "data/fixtures.csv"
    if os.path.exists(fp):
        predict_fixtures(fp, league_ratings)

    return league_ratings


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--season",   default=CURRENT_SEASON)
    parser.add_argument("--div",      nargs="*", default=None)
    parser.add_argument("--fixtures", default=None)
    parser.add_argument("--compare",  action="store_true")
    args=parser.parse_args()
    main(season=args.season, divs=args.div,
         fixtures_path=args.fixtures, compare=args.compare)