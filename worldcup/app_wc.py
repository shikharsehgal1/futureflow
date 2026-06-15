"""
app_wc.py — World Cup predictions dashboard (zero external dependencies).

A searchable reference UI for entering predictions on the SportsPredict Probability
Cup. Browse any match -> any question -> recommended probability, with the de-vigged
sharp line and the model number side by side. Search any question across all matches.

Uses only the Python standard library (no Flask needed).

Run:  python3 worldcup/app_wc.py   ->  http://localhost:5001
"""
import csv
import difflib
import json
import os
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
QUESTIONS = os.path.join(DATA, "wc_questions.csv")
SUMMARY = os.path.join(DATA, "wc_match_summary.csv")
PORT = 5001


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("prob", "pct", "pH", "pD", "pA", "total_line", "p_over",
                  "lam_h", "lam_a", "n_questions", "value", "weather_adj", "home_adv",
                  "ref_card_factor"):
            if k in r and r[k] not in ("", None):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
    return rows


# --------------------------------------------------------------------------
# Natural-language "Ask" — resolve a free-text question to the best computed row
# --------------------------------------------------------------------------
ALIASES = {
    "USA": ["usa", "united states", "america", "us"],
    "South Korea": ["korea", "south korea", "korea republic"],
    "Czech Republic": ["czechia", "czech"],
    "Bosnia & Herzegovina": ["bosnia", "herzegovina"],
    "Turkey": ["turkey", "turkiye", "türkiye"],
    "Ivory Coast": ["ivory coast", "cote d'ivoire", "côte d'ivoire", "ivoire"],
    "Curaçao": ["curacao", "curaçao"],
    "Netherlands": ["netherlands", "holland", "dutch"],
    "Saudi Arabia": ["saudi", "saudi arabia"],
    "New Zealand": ["new zealand", "nz"],
    "DR Congo": ["dr congo", "congo", "drc"],
    "South Africa": ["south africa", "rsa"],
    "Cape Verde": ["cape verde"],
}
STOP = {"will", "the", "a", "an", "be", "is", "are", "match", "at", "in", "of",
        "to", "for", "and", "by", "it", "this", "that", "do", "does", "have", "has"}
MARKET_HINTS = {
    "corner": "Corners", "corners": "Corners", "card": "Cards", "cards": "Cards",
    "booking": "Cards", "bookings": "Cards", "both": "BTTS", "btts": "BTTS",
    "win": "Moneyline", "winner": "Moneyline", "beat": "Moneyline", "draw": "Moneyline",
    "over": "Total Goals", "under": "Total Goals", "goals": "Total Goals",
    "total": "Total Goals", "half": "1st Half", "halftime": "1st Half", "ht": "1st Half",
    "scorer": "Anytime Scorer", "scores": "Anytime Scorer", "shots": "Shots on Target",
    "handicap": "Handicap", "spread": "Handicap", "double": "Double Chance",
    "chance": "Double Chance", "dnb": "Draw No Bet", "correct": "Correct Score",
    "odd": "Total Goals O/E", "even": "Total Goals O/E", "carded": "Player Cards",
}


def _norm(s):
    return re.sub(r"[^\w\s.]", " ", str(s).lower())


def _tokens(s):
    return [t for t in _norm(s).split() if t and t not in STOP]


def _team_in(q_low, team):
    for p in [team.lower()] + ALIASES.get(team, []):
        if p in q_low:
            return True
    for w in _norm(team).split():
        if len(w) >= 4 and w in q_low:
            return True
    return False


def ask_question(q):
    rows = read_csv(QUESTIONS)
    if not q or not rows:
        return {"q": q, "match": None, "answer": None, "alts": []}
    q_low = _norm(q)
    q_tok = set(_tokens(q))

    # 1. resolve the match: keep ALL matches tied on the most teams referenced,
    #    then let the question scorer + soonest kickoff break the tie.
    scores = {}
    for m in {r["match"] for r in rows}:
        teams = [t.strip() for t in m.split(" vs ")]
        scores[m] = sum(1 for t in teams if _team_in(q_low, t))
    best_score = max(scores.values()) if scores else 0
    candidates = {m for m, sc in scores.items() if sc == best_score and sc >= 1}
    pool = [r for r in rows if r["match"] in candidates] if candidates else rows

    # team tokens are already used for match resolution — strip them so they
    # don't unfairly inflate questions that merely mention the team names.
    team_tok = set()
    for m in candidates:
        for t in m.split(" vs "):
            if _team_in(q_low, t):
                team_tok |= set(_tokens(t))
                for al in ALIASES.get(t.strip(), []):
                    team_tok |= set(_tokens(al))
    score_tok = (q_tok - team_tok) or q_tok

    # 2. market hint from keywords
    hinted = {MARKET_HINTS[k] for k in MARKET_HINTS if k in q_tok}

    # 3. score every candidate question (ties broken toward the soonest kickoff)
    scored = []
    for r in pool:
        rtok = set(_tokens(r["question"]))
        cover = len(score_tok & rtok) / max(1, len(score_tok))
        ratio = difflib.SequenceMatcher(None, q_low, _norm(r["question"])).ratio()
        s = 0.65 * cover + 0.35 * ratio
        if r["market"] in hinted:
            s += 0.3
        for num in re.findall(r"\d+\.?\d*", q_low):
            if num in r["question"]:
                s += 0.2
        scored.append((s, r))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("commence", ""))))
    best_match = scored[0][1]["match"] if scored else None
    top = [{"question": r["question"], "match": r["match"], "market": r["market"],
            "pct": r["pct"], "source": r["source"], "score": round(s, 3)}
           for s, r in scored[:6]]
    return {"q": q, "match": best_match if best_score >= 1 else None,
            "answer": top[0] if top else None, "alts": top[1:]}


INDEX = r"""
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>World Cup Probability Cup — Model Dashboard</title>
<style>
 :root{--bg:#0d1117;--card:#161b22;--line:#21262d;--txt:#e6edf3;--mut:#7d8590;
       --acc:#2f81f7;--good:#3fb950;--warn:#d29922;--sharp:#a371f7}
 *{box-sizing:border-box} body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;
   background:var(--bg);color:var(--txt)}
 header{position:sticky;top:0;background:#0d1117ee;backdrop-filter:blur(8px);
   border-bottom:1px solid var(--line);padding:14px 20px;z-index:10}
 h1{font-size:17px;margin:0 0 4px} .sub{color:var(--mut);font-size:12px}
 .bar{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap}
 input,select,button{background:var(--card);border:1px solid var(--line);color:var(--txt);
   border-radius:7px;padding:8px 11px;font-size:13px;font-family:inherit}
 input{flex:1;min-width:220px} button{cursor:pointer} button:hover{border-color:var(--acc)}
 .wrap{padding:16px 20px;max-width:1150px;margin:0 auto}
 .match{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:14px;overflow:hidden}
 .mh{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;cursor:pointer}
 .mh:hover{background:#1c2330} .mt{font-weight:600;font-size:15px}
 .meta{color:var(--mut);font-size:12px;margin-top:3px}
 .pill{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;margin-left:6px}
 .soon{background:#3d1d1d;color:#ff7b72} .pre{background:#1d2d3d;color:#58a6ff}
 .wx{background:#3a2e10;color:#e8c060;margin-right:6px}
 .crowd{background:#10303a;color:#60d0e8;margin-right:6px}
 .inj{background:#3a1020;color:#ff8088;margin-right:6px} .s-skip{background:#3a2122;color:#d9a}
 .ref{background:#2a2440;color:#c8b0f0;margin-right:6px}
 .body{display:none;border-top:1px solid var(--line);padding:6px 0}
 .body.open{display:block}
 .grp{padding:6px 16px} .gname{color:var(--mut);font-size:11px;text-transform:uppercase;
   letter-spacing:.05em;margin:8px 0 4px}
 .row{display:flex;align-items:center;justify-content:space-between;padding:6px 8px;border-radius:6px}
 .row:hover{background:#1c2330} .q{flex:1}
 .src{font-size:10px;padding:1px 6px;border-radius:4px;margin-left:8px;color:#fff}
 .s-sharp{background:var(--sharp)} .s-poisson{background:#30506b} .s-consensus{background:#5a4a1a}
 .s-heuristic{background:#444c56} .s-elo{background:#1f6f4a}
 .val{font-weight:700;font-size:15px;min-width:62px;text-align:right;cursor:pointer}
 .val:hover{color:var(--acc)} .hi{color:var(--good)} .lo{color:var(--mut)}
 .copied{color:var(--good);font-size:11px;margin-left:6px}
 .legend{color:var(--mut);font-size:11px;margin-top:8px}
 .count{color:var(--mut);font-size:12px}
 .askbar{display:flex;gap:10px;margin-top:10px}
 #ask{flex:1;border-color:var(--acc)} .askbtn{background:var(--acc);border-color:var(--acc);font-weight:600}
 #answer{margin-top:10px}
 .ansbox{background:#11203a;border:1px solid var(--acc);border-radius:10px;padding:14px 16px}
 .ansrow{display:flex;align-items:center;justify-content:space-between;gap:14px}
 .anstitle{font-size:15px;font-weight:600} .ansmeta{color:var(--mut);font-size:12px;margin-top:3px}
 .ansval{font-size:30px;font-weight:800;color:var(--good);cursor:pointer;white-space:nowrap}
 .ansval:hover{color:#56d364} .anshint{color:var(--mut);font-size:11px;text-align:right}
 .alts{margin-top:10px;border-top:1px solid var(--line);padding-top:8px;display:flex;flex-wrap:wrap;gap:6px}
 .alt{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 9px;
   font-size:12px;cursor:pointer} .alt:hover{border-color:var(--acc)}
 .altpct{font-weight:700;color:var(--txt)}
 .tier{font-size:9px;padding:1px 5px;border-radius:4px;margin-left:6px;font-weight:700;letter-spacing:.03em}
 .t-HIGH{background:#1f6f4a;color:#fff} .t-MEDIUM{background:#2a3b4d;color:#9fc} .t-SKIP{background:#3a2122;color:#d9a}
 .row.skip{opacity:.5}
</style></head><body>
<header>
 <h1>🏆 World Cup — Probability Cup Model Dashboard</h1>
 <div class="sub">De-vigged sharp (Pinnacle) + Poisson model. Click a % to copy. Enter your final value just before kickoff.</div>
 <div class="askbar">
   <input id="ask" placeholder="🎯 Ask a question… e.g. 'will Brazil win', 'over 2.5 goals Korea Czechia', 'Son to score', 'corners over 9.5'">
   <button class="askbtn" onclick="ask()">Ask</button>
 </div>
 <div id="answer"></div>
 <div class="bar">
   <input id="search" placeholder="🔍 Filter the list below… (team, market, or question text)">
   <select id="mkt"><option value="">All markets</option></select>
   <select id="tier"><option value="">All tiers</option><option>HIGH</option><option>MEDIUM</option><option>SKIP</option></select>
   <select id="sort"><option value="time">Sort: kickoff</option><option value="alpha">Sort: A–Z</option><option value="value">Sort: value</option></select>
   <button onclick="refresh(false)">↻ Recompute</button>
   <button onclick="refresh(true)" title="Re-pull odds (uses API credits)">↻ Re-pull odds</button>
 </div>
 <div class="legend"><span class="count" id="count"></span> &nbsp;·&nbsp;
   <span class="src s-sharp">sharp</span> direct de-vigged Pinnacle &nbsp;
   <span class="src s-poisson">poisson</span> goal model &nbsp;
   <span class="src s-consensus">consensus</span> book median &nbsp;
   <span class="src s-heuristic">heuristic</span></div>
</header>
<div class="wrap" id="app"></div>
<script>
let QS=[], INJ={};
const grp=(a,k)=>a.reduce((m,x)=>((m[x[k]]=m[x[k]]||[]).push(x),m),{});
function fmtCountdown(iso){
  const ms=new Date(iso)-new Date(); if(ms<0)return{t:'LIVE/DONE',c:'soon'};
  const h=Math.floor(ms/3.6e6), m=Math.floor(ms%3.6e6/6e4);
  const s=h<6?'soon':'pre';
  return{t:(h>=24?Math.floor(h/24)+'d ':'')+(h%24)+'h '+m+'m',c:s};
}
async function load(){
  QS=await (await fetch('/api/questions')).json();
  try{const inj=await (await fetch('/api/injuries')).json();
      INJ={}; inj.forEach(r=>{(INJ[r.team]=INJ[r.team]||[]).push(r);});}catch(e){INJ={};}
  const mkts=[...new Set(QS.map(q=>q.market))].sort();
  document.getElementById('mkt').innerHTML='<option value="">All markets</option>'+
    mkts.map(m=>`<option>${m}</option>`).join('');
  render();
}
function render(){
  const term=document.getElementById('search').value.toLowerCase().trim();
  const mf=document.getElementById('mkt').value;
  const tf=document.getElementById('tier').value;
  const sort=document.getElementById('sort').value;
  let qs=QS.filter(q=>(!mf||q.market===mf)&&(!tf||q.tier===tf)&&
     (!term||(q.match+' '+q.question+' '+q.market).toLowerCase().includes(term)));
  document.getElementById('count').textContent=qs.length+' questions';
  let matches=grp(qs,'match');
  let order=Object.keys(matches);
  if(sort==='alpha')order.sort();
  else if(sort==='value')order.sort((a,b)=>Math.max(...matches[b].map(x=>+x.value||0))-Math.max(...matches[a].map(x=>+x.value||0)));
  else order.sort((a,b)=>new Date(matches[a][0].commence)-new Date(matches[b][0].commence));
  const open = term||mf||tf;
  document.getElementById('app').innerHTML=order.map(mn=>{
    const rows=matches[mn], cd=fmtCountdown(rows[0].commence);
    const teams=mn.split(' vs ');
    const inj=teams.flatMap(t=>(INJ[t]||[]).map(r=>({...r,_team:t})));
    const injBadge=inj.length?`<span class="pill inj" title="${inj.map(r=>r._team+': '+r.player+' ('+r.status+')').join(' · ')}">🚑 ${inj.length}</span>`:'';
    const injLine=inj.length?`<div class="grp"><div class="gname">🚑 Injuries / late news (already mostly priced — flag for fresh breaks)</div>`+
      inj.map(r=>`<div class="row"><span class="q">${r._team}: <b>${r.player}</b> <span class="src s-skip">${r.status}</span> <span style="color:var(--mut);font-size:11px">${r.note||''}</span></span></div>`).join('')+`</div>`:'';
    const byMkt=grp(rows,'market');
    const inner=Object.keys(byMkt).sort().map(g=>`<div class="grp"><div class="gname">${g}</div>`+
      (sort==='value'?byMkt[g].slice().sort((a,b)=>(+b.value||0)-(+a.value||0)):byMkt[g]).map(q=>{
        const cls='s-'+(String(q.source).split(':')[0]);
        const vc=q.pct>=60?'hi':(q.pct<=40?'lo':'');
        return `<div class="row${q.tier==='SKIP'?' skip':''}"><span class="q">${q.question}
          <span class="src ${cls}">${q.source}</span><span class="tier t-${q.tier}">${q.tier}</span></span>
          <span class="val ${vc}" onclick="cp(this,${q.pct})">${q.pct}%</span></div>`;
      }).join('')+`</div>`).join('');
    return `<div class="match"><div class="mh" onclick="this.nextElementSibling.classList.toggle('open')">
      <div><div class="mt">${mn}</div><div class="meta">${rows.length} questions · ${new Date(rows[0].commence).toLocaleString()}</div></div>
      <div>${(rows[0].referee && Math.abs(+rows[0].ref_card_factor-1)>=0.05)?`<span class="pill ref" title="referee ${rows[0].referee} — cards run ${rows[0].ref_card_factor}x the field average">⚖️ ${rows[0].referee.split(' ').pop()} ${rows[0].ref_card_factor}x</span>`:''}${injBadge}${(Math.abs(+rows[0].home_adv)>=0.1)?`<span class="pill crowd" title="diaspora/host crowd lean (fundamentals factor; market already prices it)">🏟️ ${(+rows[0].home_adv>0?'+':'')}${rows[0].home_adv}</span>`:''}${(+rows[0].weather_adj<0.92)?`<span class="pill wx" title="goal-suppressing conditions (heat/wind/rain)">🌡️ ${rows[0].weather_adj}</span>`:''}<span class="pill ${cd.c}">${cd.t}</span></div></div>
      <div class="body ${open?'open':''}">${injLine}${inner}</div></div>`;
  }).join('')||'<p class="sub">No questions match.</p>';
}
function cp(el,v){navigator.clipboard.writeText(v);const s=document.createElement('span');
  s.className='copied';s.textContent='copied '+v;el.after(s);setTimeout(()=>s.remove(),1200);}
async function ask(){
  const q=document.getElementById('ask').value.trim(); if(!q)return;
  const box=document.getElementById('answer'); box.innerHTML='<div class="ansbox">thinking…</div>';
  const r=await (await fetch('/api/ask?q='+encodeURIComponent(q))).json();
  if(!r.answer){box.innerHTML='<div class="ansbox">No matching question found. Try naming both teams + the market.</div>';return;}
  const A=r.answer, cls='s-'+String(A.source).split(':')[0];
  const alts=r.alts.map(x=>`<span class="alt" title="${x.match} · ${x.question}" onclick="cp(this,${x.pct})">
     <span class="altpct">${x.pct}%</span> ${x.market}: ${x.question.replace(/^Will /,'').replace(/\?$/,'')}</span>`).join('');
  box.innerHTML=`<div class="ansbox">
     <div class="ansrow"><div>
        <div class="anstitle">${A.question}</div>
        <div class="ansmeta">${A.match} · ${A.market} · <span class="src ${cls}">${A.source}</span></div>
     </div><div style="text-align:right"><div class="ansval" onclick="cp(this,${A.pct})">${A.pct}%</div>
        <div class="anshint">click to copy</div></div></div>
     ${alts?`<div class="alts">${alts}</div>`:''}</div>`;
}
async function refresh(fetchOdds){
  document.getElementById('count').textContent='recomputing…';
  const r=await (await fetch('/api/refresh'+(fetchOdds?'?fetch=1':''),{method:'POST'})).json();
  if(r.ok)load();else alert('Error: '+r.error);
}
['search','mkt','tier','sort'].forEach(id=>document.getElementById(id).addEventListener('input',render));
document.getElementById('ask').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
load();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, INDEX, "text/html; charset=utf-8")
        if path == "/api/questions":
            return self._send(200, read_csv(QUESTIONS))
        if path == "/api/summary":
            return self._send(200, read_csv(SUMMARY))
        if path == "/api/players":
            return self._send(200, read_csv(os.path.join(DATA, "player_rankings.csv")))
        if path == "/api/injuries":
            return self._send(200, read_csv(os.path.join(DATA, "wc_injuries.csv")))
        if path == "/api/elo":
            return self._send(200, read_csv(os.path.join(DATA, "wc_elo_questions.csv")))
        if path == "/api/fundamentals":   # lbenz bivariate-Poisson + crowd + momentum
            return self._send(200, read_csv(os.path.join(DATA, "wc_fundamentals.csv")))
        if path == "/api/sim":            # tournament sim: advance/reach-round/champion probs
            return self._send(200, read_csv(os.path.join(DATA, "wc_sim_results.csv")))
        if path == "/api/ask":
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0]
            return self._send(200, ask_question(q))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/refresh":
            q = parse_qs(parsed.query)
            try:
                if q.get("fetch", ["0"])[0] == "1":
                    subprocess.run([sys.executable, os.path.join(HERE, "fetch_wc_odds.py")],
                                   check=True, cwd=HERE, timeout=120)
                subprocess.run([sys.executable, os.path.join(HERE, "predict_wc.py")],
                               check=True, cwd=HERE, timeout=180)
                return self._send(200, {"ok": True, "n": len(read_csv(QUESTIONS))})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)})
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"World Cup dashboard -> http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
