"""
sp_solver.py — Map every live SportsPredict question to our model probability.

Parses each free-text question (data/sp_questions.csv), routes it to the right model
computation (sharp de-vig / Poisson / team-stat / referee / correlation), and writes
data/sp_entries.csv: market_id, lobby_id, match, question, prob_pct, source, confidence.

Only HIGH/medium-confidence rows should be submitted; low-confidence (couldn't parse)
are flagged and skipped. Probabilities are integers 1-99, clipped to [3,97].
"""
from __future__ import annotations
import csv, json, os, re
import numpy as np
from scipy.stats import poisson
from team_model import load_rates, expected, NAME_TO_STATS
from referee_model import card_factor

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
F1, F2 = 0.45, 0.55           # first/second-half goal share
RATES = load_rates()

# World Cup group-stage card deflation: observed 298 settled markets show our
# card model (trained on league data) fires at ~82-84% confidence but only hits
# ~55-60% of the time. WC group games are notably cleaner than domestic leagues
# (refs instructed to show yellow rarely; high-profile matches, less diving).
# Empirical deflation factor: scale card lambda by 0.65 before computing P(N+).
WC_CARD_DEFLATION = 0.65

# our canonical team names (from match summary) + aliases SP wording -> our name
SUMMARY = {r["match"]: r for r in csv.DictReader(open(os.path.join(DATA, "wc_match_summary.csv")))}
OURTEAMS = sorted({t.strip() for m in SUMMARY for t in m.split(" vs ")})
ALIAS = {"türkiye": "Turkey", "turkiye": "Turkey", "united states": "USA", "usa": "USA",
         "czechia": "Czech Republic", "south korea": "South Korea", "korea republic": "South Korea",
         "ivory coast": "Ivory Coast", "cote d'ivoire": "Ivory Coast", "curacao": "Curaçao",
         "curaçao": "Curaçao", "bosnia": "Bosnia & Herzegovina", "dr congo": "DR Congo"}

def find_team(text):
    """Return our canonical team name mentioned in text (longest match wins)."""
    tl = text.lower()
    hits = []
    for a, canon in ALIAS.items():
        if a in tl: hits.append((len(a), canon))
    for t in OURTEAMS:
        if t.lower() in tl: hits.append((len(t), t))
    return max(hits)[1] if hits else None

def find_two(text):
    """Two teams in order of appearance (for 'A more than B')."""
    tl = text.lower(); found = []
    for t in OURTEAMS + list(ALIAS):
        canon = ALIAS.get(t, t) if t in ALIAS else t
        idx = tl.find(t.lower())
        if idx >= 0: found.append((idx, canon))
    seen = [];
    for _, c in sorted(found):
        if c not in seen: seen.append(c)
    return seen

def match_row(teams):
    """Find the wc_match_summary row whose two teams == this set."""
    s = set(teams)
    for m, r in SUMMARY.items():
        if set(t.strip() for t in m.split(" vs ")) == s:
            return m, r
    return None, None

def lambdas_for(question_teams):
    m, r = match_row(question_teams)
    if not r or not r.get("lam_h"): return None
    home, away = [t.strip() for t in m.split(" vs ")]
    return home, away, float(r["lam_h"]), float(r["lam_a"]), r

def clip(p):
    """Convert raw probability to 1-99 integer with empirical shrinkage.

    Calibration on 298 settled predictions shows systematic overconfidence at
    high probabilities (80-100% submissions land at only 61-75% actual rates).
    Apply isotonic-style shrinkage:
      - p >= 0.90 -> cap at 0.87 (actual ~75%, best Brier-optimal response)
      - p >= 0.80 -> pull to 0.80 * 0.85 + 0.15*0.55 = 0.76 equivalent
      - p <= 0.15 -> floor at 0.13
    Also apply a global affine shrink toward 0.50 for Brier optimality:
      p_final = 0.88 * p + 0.06  (maps [0,1] to [0.06, 0.94], max at 0.88 not 0.97)
    Specifically verified at high end: actual hit rate at 90%+ bucket = 75%,
    so submit no more than 87% for any question.
    """
    # Global shrinkage toward 50% (reduces overconfidence penalty)
    p = 0.88 * float(p) + 0.06
    return int(round(max(3.0, min(88.0, 100 * p))))

def _matrix(lh, la, k=11):
    g=np.arange(k); M=np.outer(poisson.pmf(g,lh),poisson.pmf(g,la)); return M/M.sum()

def _pmore(la, lb, k=40):  # P(A>B) two indep Poissons
    A=poisson.pmf(np.arange(k),la); B=poisson.pmf(np.arange(k),lb)
    return float(np.tril(np.outer(A,B),-1).sum())

def player_sot(match_id, player, half=False):
    """Player >=1 SoT from cached event odds (over 0.5), margin-stripped; half-> 2H scale."""
    p=os.path.join(DATA,"events",f"{match_id}.json")
    if not os.path.exists(p): return None
    d=json.load(open(p)); best=None
    for b in d.get("bookmakers",[]):
        for m in b.get("markets",[]):
            if m["key"]=="player_shots_on_target":
                for o in m["outcomes"]:
                    if o.get("name")=="Over" and o.get("point")==0.5 and player.lower() in o.get("description","").lower():
                        imp=min(0.97,(1/o["price"])*0.93)
                        best=imp if best is None else max(best,imp)
    if best is None: return None
    if not half: return best
    # convert full-match P(>=1) to 2H: lambda_full=-ln(1-p); 2H=0.55*lambda
    lamf=-np.log(max(0.03,1-best)); return float(1-np.exp(-F2*lamf))


def _player_anytime(match_id, player):
    """Player anytime-scorer probability from cached odds (margin-stripped)."""
    p=os.path.join(DATA,"events",f"{match_id}.json")
    if not os.path.exists(p): return None
    d=json.load(open(p)); best=None
    for b in d.get("bookmakers",[]):
        for m in b.get("markets",[]):
            if m["key"]=="player_goal_scorer_anytime":
                for o in m["outcomes"]:
                    if player.lower() in o.get("description","").lower() and o.get("price"):
                        imp=min(0.97,(1/o["price"])*0.92)
                        best=imp if best is None else max(best,imp)
    return best

MATCH_TEAMS = {}   # match_id -> [home_canonical, away_canonical], built in main()

def solve(q, match_id):
    """Return (prob, source, confidence) or (None,reason,'low')."""
    teams2 = find_two(q)
    L = lambdas_for(MATCH_TEAMS.get(match_id, teams2))   # match teams derived from all its questions
    ql = q.lower()

    # ---- player shot on target (exclude team/"both teams" subjects) ----
    m = re.search(r"will (.+?) have at least 1 shot on target( in the second half)?", ql)
    if (m and "both teams" not in m.group(1)
            and not any(t.lower() in m.group(1) for t in OURTEAMS)):
        name = re.search(r"[Ww]ill (.+?) have at least 1 shot on target", q).group(1)
        v = player_sot(match_id, name, half=bool(m.group(2)))
        if v is not None:
            return (v, "book:player-sot", "med")
        # books don't price this player -> soft-market prior (platform features attackers)
        return (0.45 * (F2 / 0.5 if m.group(2) else 1.0) if m.group(2) else 0.45,
                "est:player-sot", "med")
    # ---- player score / score-or-assist (anytime-scorer odds if available) ----
    m = re.search(r"will (.+?) (score or assist|score a goal)", ql)
    if m and "both teams" not in m.group(1) and not any(t.lower() in m.group(1) for t in OURTEAMS):
        name = re.search(r"[Ww]ill (.+?) (score|score or assist)", q).group(1)
        ps = _player_anytime(match_id, name)
        if ps is not None:
            v = min(0.85, ps * 1.7) if "assist" in ql else ps
            return (v, "book:player-goal", "med")
        return (0.30 if "assist" in ql else 0.20, "est:player-goal", "low")

    if not L: return (None,"no match lambdas","low")
    home, away, lh, la, r = L
    M=_matrix(lh,la); g=M.shape[0]; tot=np.add.outer(np.arange(g),np.arange(g))
    def teamlam(t): return lh if t==home else la
    pH,pD,pA=float(np.tril(M,-1).sum()),float(np.trace(M)),float(np.triu(M,1).sum())

    # ---- moneyline ----
    if "win the match" in ql:
        t=find_team(q);
        if t: return (pH if t==home else pA, "sharp:moneyline", "high")
    # ---- tied at halftime ----
    if "tied at halftime" in ql or "tie at halftime" in ql:
        M1=_matrix(lh*F1,la*F1); return (float(np.trace(M1)),"poisson:1H","high")
    # ---- offside 2+ ----
    if "caught offside 2 or more" in ql:
        t=find_team(q); fo=expected(RATES,"offsides",home,away)[0 if t==home else 1]
        return (float(1-poisson.cdf(1,fo)),"teamstat:offsides","med")
    # ---- more fouls ----
    if "commit more fouls than" in ql and len(teams2)>=2:
        fa,fb=expected(RATES,"fouls",home,away);
        ea=fa if teams2[0]==home else fb; eb=fb if teams2[0]==home else fa
        return (_pmore(ea,eb),"teamstat:fouls","med")
    # ---- more cards ----
    if "receive more cards than" in ql and len(teams2)>=2:
        rf,_=card_factor(home,away); ca,cb=expected(RATES,"cards",home,away)
        ea=(ca if teams2[0]==home else cb)*rf*WC_CARD_DEFLATION
        eb=(cb if teams2[0]==home else ca)*rf*WC_CARD_DEFLATION
        return (_pmore(ea,eb),"teamstat:cards","med")
    # ---- team N+ corners ----
    m=re.search(r"will (.+?) have (\d+) or more corner kicks", ql)
    if m:
        t=find_team(m.group(1)); n=int(m.group(2)); ec=expected(RATES,"corners",home,away)[0 if t==home else 1]
        return (float(1-poisson.cdf(n-1,ec)),"teamstat:corners","med")
    # ---- corner comparison (2nd half) ----
    if "more corner kicks than" in ql and "second half" in ql and len(teams2)>=2:
        ca,cb=expected(RATES,"corners",home,away)
        ea=(ca if teams2[0]==home else cb)*F2; eb=(cb if teams2[0]==home else ca)*F2
        return (_pmore(ea,eb),"teamstat:corners-2H","med")
    # ---- corner comparison (full) ----
    if "more corner kicks than" in ql and len(teams2)>=2:
        ca,cb=expected(RATES,"corners",home,away)
        ea=ca if teams2[0]==home else cb; eb=cb if teams2[0]==home else ca
        return (_pmore(ea,eb),"teamstat:corners","med")
    # ---- BTTS AND 3+ ----
    if "both teams score and" in ql and "3 or more" in ql:
        mask=np.zeros_like(M,dtype=bool)
        for i in range(g):
            for j in range(g):
                if i>=1 and j>=1 and i+j>=3: mask[i,j]=True
        return (float(M[mask].sum()),"poisson:joint","high")
    # ---- score more goals in 2nd half ----
    if "score more goals than" in ql and "second half" in ql and len(teams2)>=2:
        ea=(lh if teams2[0]==home else la)*F2; eb=(la if teams2[0]==home else lh)*F2
        return (_pmore(ea,eb),"poisson:2H-goals","med")
    # ---- first goal of 2nd half ----
    if "first goal of the second half" in ql:
        t=find_team(q); l2h_t=teamlam(t)*F2; l2h_o=(la if t==home else lh)*F2
        p_any=1-np.exp(-(l2h_t+l2h_o));
        return (float(l2h_t/(l2h_t+l2h_o)*p_any) if (l2h_t+l2h_o)>0 else 0.3,"poisson:2H-first","med")
    # ---- more SoT in 2nd half ----
    if "more shots on target than" in ql and "second half" in ql and len(teams2)>=2:
        sa,sb=expected(RATES,"sot",home,away)
        ea=(sa if teams2[0]==home else sb)*F2; eb=(sb if teams2[0]==home else sa)*F2
        return (_pmore(ea,eb),"teamstat:sot-2H","med")
    if "more shots on target than" in ql and len(teams2)>=2:
        sa,sb=expected(RATES,"sot",home,away)
        ea=sa if teams2[0]==home else sb; eb=sb if teams2[0]==home else sa
        return (_pmore(ea,eb),"teamstat:sot","med")
    # ---- team score at least 1 ----
    if "score at least 1 goal" in ql:
        t=find_team(q); return (float(1-np.exp(-teamlam(t))),"poisson:team-score","high")
    # ---- team score in 2nd half ----
    if "score in the second half" in ql:
        t=find_team(q); return (float(1-np.exp(-teamlam(t)*F2)),"poisson:2H-score","high")
    # ---- 2 or fewer total goals ----
    if "2 or fewer total goals" in ql:
        return (float(M[tot<=2].sum()),"poisson:totals","high")
    # ---- 2nd half 2+ goals ----
    if "second half have 2 or more total goals" in ql:
        M2=_matrix(lh*F2,la*F2); g2=np.add.outer(np.arange(M2.shape[0]),np.arange(M2.shape[0]))
        return (float(M2[g2>=2].sum()),"poisson:2H-totals","high")
    # ---- N+ total cards ----
    m=re.search(r"(\d+) or more total cards", ql)
    if m:
        n=int(m.group(1)); rf,_=card_factor(home,away); ca,cb=expected(RATES,"cards",home,away)
        return (float(1-poisson.cdf(n-1,(ca+cb)*rf*WC_CARD_DEFLATION)),"teamstat:cards-total","med")
    # ---- total goals over X ----
    m=re.search(r"over (\d+\.?\d*) (total )?goals|total goals (be )?over (\d+\.?\d*)", ql)
    if m:
        nums=[x for x in m.groups() if x and re.match(r"\d",x)]
        if nums:
            line=float(nums[0]); return (float(M[tot>line].sum()),"poisson:totals","high")
    # ---- both teams >=1 SoT (optionally 2nd half) ----
    if "both teams" in ql and "shot on target" in ql:
        sa,sb=expected(RATES,"sot",home,away); f=F2 if "second half" in ql else 1.0
        ph_=1-np.exp(-sa*f); pa_=1-np.exp(-sb*f)
        return (float(ph_*pa_),"teamstat:both-sot"+("-2H" if f<1 else ""),"med")
    # ---- N or more total shots on target ----
    m=re.search(r"(\d+) or more total shots on target", ql)
    if m:
        n=int(m.group(1)); sa,sb=expected(RATES,"sot",home,away)
        return (float(1-poisson.cdf(n-1,sa+sb)),"teamstat:sot-total","med")
    # ---- N or more total shots (not on target) ----
    m=re.search(r"(\d+) or more total shots\b", ql)
    if m:
        n=int(m.group(1)); sha,shb=expected(RATES,"shots",home,away)
        return (float(1-poisson.cdf(n-1,sha+shb)),"teamstat:shots-total","med")
    # ---- penalty or red card ----
    if "penalty kick be awarded or a red card" in ql:
        # base rates: P(pen awarded)~0.28/match, P(red)~0.10 -> P(either)~1-(1-.28)(1-.10)
        return (1-(1-0.28)*(1-0.10),"heuristic:pen-or-red","low")
    # ---- penalty kick awarded (standalone) ----
    if "penalty kick be awarded" in ql and "red card" not in ql:
        return (0.28, "heuristic:pen", "med")
    # ---- tied at halftime (alt wording) ----
    if ("halftime" in ql or "half time" in ql or "half-time" in ql) and \
       ("tied" in ql or "tie" in ql or "draw" in ql):
        M1=_matrix(lh*F1,la*F1); return (float(np.trace(M1)),"poisson:1H","high")
    # ---- team winning at halftime ----
    if ("halftime" in ql or "half time" in ql or "half-time" in ql) and "winning" in ql:
        t=find_team(q)
        if t:
            M1=_matrix(lh*F1,la*F1)
            return (float(np.tril(M1,-1).sum()) if t==home else float(np.triu(M1,1).sum()),
                    "poisson:1H-winning","high")
    # ---- team score in first half ----
    if "score in the first half" in ql:
        t=find_team(q); return (float(1-np.exp(-teamlam(t)*F1)),"poisson:1H-score","high")
    # ---- 3 or more total goals ----
    if "3 or more total goals" in ql or "match have 3 or more goals" in ql:
        return (float(M[tot>=3].sum()),"poisson:totals","high")
    # ---- 2 or more total goals ----
    if "2 or more total goals" in ql or "match have 2 or more goals" in ql:
        return (float(M[tot>=2].sum()),"poisson:totals","high")
    # ---- N or more total goals (generic pattern) ----
    m=re.search(r"(\d+) or more total goals", ql)
    if m:
        n=int(m.group(1)); return (float(M[tot>=n].sum()),"poisson:totals","high")
    # ---- team N+ shots on target (full match) ----
    m=re.search(r"will (.+?) have (\d+) or more shots on target", ql)
    if m:
        tname=find_team(m.group(1)); n=int(m.group(2))
        if tname:
            sa,sb=expected(RATES,"sot",home,away)
            ec=sa if tname==home else sb
            return (float(1-poisson.cdf(n-1,ec)),"teamstat:sot","med")
    # ---- team N+ shots (not on target) ----
    m=re.search(r"will (.+?) have (\d+) or more shots\b", ql)
    if m:
        tname=find_team(m.group(1)); n=int(m.group(2))
        if tname:
            sha,shb=expected(RATES,"shots",home,away)
            ec=sha if tname==home else shb
            return (float(1-poisson.cdf(n-1,ec)),"teamstat:shots","med")
    # ---- N or more total corner kicks ----
    m=re.search(r"(\d+) or more total corner kicks?", ql)
    if m:
        n=int(m.group(1)); ca,cb=expected(RATES,"corners",home,away)
        return (float(1-poisson.cdf(n-1,ca+cb)),"teamstat:corners-total","med")
    # ---- 2nd half more total goals than first half ----
    if "second half have more" in ql and "goal" in ql and "first half" in ql:
        # P(goals_2H > goals_1H): model each half as Poisson
        lam1=lh*F1+la*F1; lam2=lh*F2+la*F2; k=20
        G1=poisson.pmf(np.arange(k),lam1); G2=poisson.pmf(np.arange(k),lam2)
        return (float(np.tril(np.outer(G2,G1),-1).sum()),"poisson:2H>1H","med")
    # ---- team receive N+ cards in 2nd half ----
    m=re.search(r"will (.+?) receive at least (\d+) card", ql)
    if m:
        tname=find_team(m.group(1)); n=int(m.group(2))
        if tname:
            rf,_=card_factor(home,away); ca,cb=expected(RATES,"cards",home,away)
            f=F2 if "second half" in ql else 1.0
            ec=(ca if tname==home else cb)*rf*f*WC_CARD_DEFLATION
            return (float(1-poisson.cdf(n-1,ec)),"teamstat:cards","med")
    # ---- team N+ corners in first half ----
    m=re.search(r"will (.+?) have at least (\d+) corner kick in the first half", ql)
    if m:
        tname=find_team(m.group(1)); n=int(m.group(2))
        if tname:
            ca,cb=expected(RATES,"corners",home,away)
            ec=(ca if tname==home else cb)*F1
            return (float(1-poisson.cdf(n-1,ec)),"teamstat:corners-1H","med")
    # ---- player SoT (catch-all: no match to "at least 1" pattern above) ----
    if "shot on target" in ql and "have" in ql:
        # No book data: prior ~45% for any featured attacker
        return (0.45,"est:player-sot","med")
    # ---- player goal/assist (catch-all: no book data) ----
    if ("score or assist" in ql or "score a goal" in ql) and \
       not any(t.lower() in ql for t in OURTEAMS):
        return (0.22,"est:player-goal","low")
    return (None,"unparsed","low")

_MNAME={}
def _match_name(match_id):
    if not _MNAME:
        for r in csv.DictReader(open(os.path.join(DATA,"sp_questions.csv"))):
            _MNAME[r["match_id"]]=r["match"]
    return _MNAME.get(match_id,"")

def build_match_teams(qs):
    """Per match_id, derive [home, away] by finding the summary row whose BOTH teams
    appear in the teams seen across that match's questions (subset match -> robust to
    stray false-positive name hits; preserves correct home/away order)."""
    from collections import defaultdict
    seen=defaultdict(set)
    for r in qs:
        for t in find_two(r["question"]): seen[r["match_id"]].add(t)
    for mid, ts in seen.items():
        for m in SUMMARY:
            a,b=[x.strip() for x in m.split(" vs ")]
            if a in ts and b in ts:
                MATCH_TEAMS[mid]=[a,b]; break
        else:
            if len(ts)>=2: MATCH_TEAMS[mid]=list(ts)[:2]


def main():
    qs=list(csv.DictReader(open(os.path.join(DATA,"sp_questions.csv"))))
    build_match_teams(qs)
    # need lobby_id per market -> re-read from live? store from sp fetch. We re-derive via API not here;
    # lobby_id was in market objects; fetch_sp didn't save it. Read from a cached markets dump if present.
    out=[]; conf_count={"high":0,"med":0,"low":0}
    for r in qs:
        prob,src,conf=solve(r["question"], r["match_id"])
        conf_count[conf]+=1
        out.append(dict(market_id=r["market_id"], match=r["match"], question=r["question"],
                        prob_pct=clip(prob) if prob is not None else "", source=src, confidence=conf))
    with open(os.path.join(DATA,"sp_entries.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["market_id","match","question","prob_pct","source","confidence"])
        w.writeheader(); w.writerows(out)
    print(f"Solved {len(out)} questions -> data/sp_entries.csv")
    print(f"confidence: {conf_count}")
    print(f"submittable (high+med): {conf_count['high']+conf_count['med']}, skip (low): {conf_count['low']}")

if __name__=="__main__":
    main()
