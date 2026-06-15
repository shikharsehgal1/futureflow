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

def clip(p): return int(round(max(3.0, min(97.0, 100*p))))

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
        ea=(ca if teams2[0]==home else cb)*rf; eb=(cb if teams2[0]==home else ca)*rf
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
        return (float(1-poisson.cdf(n-1,(ca+cb)*rf)),"teamstat:cards-total","med")
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
