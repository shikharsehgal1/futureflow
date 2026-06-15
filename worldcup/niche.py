"""
niche.py — Parse per-event Odds API markets into platform questions.

Handles the markets that the cheap featured endpoint doesn't carry: BTTS, double
chance, draw-no-bet, 1st-half result/totals, team totals, total corners, total
cards, anytime goalscorer, player shots-on-target, player to be carded.

De-vig rules:
  * 2-way / 3-way exhaustive markets (btts, dc, dnb, h2h_h1, totals, corners,
    cards, team_totals) -> proper multiplicative de-vig (probabilities sum to 1).
  * Player "Yes"-only markets (anytime scorer, to-receive-card, shots) are NOT
    exhaustive (many players can score), so they can't be normalised. We strip a
    flat book margin instead — tighter for Pinnacle, looser elsewhere.

For every market we prefer Pinnacle (sharpest); if absent we fall back to the
median de-vigged value across all books that priced it.
"""
from __future__ import annotations
import numpy as np
from devig import devig, devig_two, clip_prob

BOOK_PREF = ["pinnacle", "matchbook", "betfair_ex_eu", "williamhill", "onexbet"]
# margin factor applied to Yes-only player props (P_fair ≈ implied * factor)
PROP_MARGIN = {"pinnacle": 0.955, "matchbook": 0.95, "betfair_ex_eu": 0.97}
PROP_MARGIN_DEFAULT = 0.92


def _markets_by_book(njson):
    out = {}
    for b in njson.get("bookmakers", []):
        out[b["key"]] = {m["key"]: m for m in b.get("markets", [])}
    return out


def _ordered_books(mbb, key):
    have = [bk for bk in mbb if key in mbb[bk]]
    pref = [bk for bk in BOOK_PREF if bk in have]
    return pref + [bk for bk in have if bk not in pref]


def _q(rows, match, mkt, question, prob, source, note=""):
    if prob is None or not np.isfinite(prob):
        return
    rows.append(dict(match=match, market=mkt, question=question,
                     prob=round(clip_prob(prob), 4), pct=round(100 * clip_prob(prob), 1),
                     source=source, note=note))


def niche_questions(njson, mp):
    home, away = mp["home"], mp["away"]
    match = f"{home} vs {away}"
    mbb = _markets_by_book(njson)
    rows = []

    def best(key, extract):
        """extract(outcomes_list) -> prob or None. Prefer pinnacle, else median."""
        vals, used_book = [], None
        for bk in _ordered_books(mbb, key):
            v = extract(mbb[bk][key]["outcomes"])
            if v is not None and np.isfinite(v):
                vals.append(v)
                if used_book is None:
                    used_book = bk
        if not vals:
            return None, None
        if used_book == "pinnacle":
            return vals[0], "sharp:pinnacle"
        return float(np.median(vals)), f"consensus:{used_book or 'median'}"

    def price(outs, **match_kw):
        for o in outs:
            if all(o.get(k) == v for k, v in match_kw.items()):
                return o["price"]
        return None

    # ---- BTTS (Yes/No) ----
    def btts_yes(outs):
        y, n = price(outs, name="Yes"), price(outs, name="No")
        return devig_two(y, n) if y and n else None
    p, s = best("btts", btts_yes)
    _q(rows, match, "BTTS", "Will both teams score?", p, s)
    if p is not None:
        _q(rows, match, "BTTS", "Will NOT both teams score?", 1 - p, s)

    # ---- Double chance (3-way) ----
    for sel, label in ((f"{home} or Draw", f"{home} or Draw (1X)"),
                       (f"{away} or Draw", f"{away} or Draw (X2)"),
                       (f"{home} or {away}", f"{home} or {away} (12)")):
        def dc(outs, sel=sel):
            ps = [o["price"] for o in outs]
            names = [o["name"] for o in outs]
            if sel not in names or len(ps) != 3:
                return None
            # Double-chance outcomes each cover 2 of 3 results, so fair probs
            # sum to 2.0 (not 1.0). De-vig by normalising the overround to 2.
            raw = np.array([1.0 / p for p in ps])
            raw = raw * (2.0 / raw.sum())
            return float(raw[names.index(sel)])
        p, s = best("double_chance", dc)
        _q(rows, match, "Double Chance", label, p, s)

    # ---- Draw no bet ----
    def dnb(outs, team):
        o = {x["name"]: x["price"] for x in outs}
        if home in o and away in o:
            return devig_two(o[team], o[away if team == home else home])
        return None
    p, s = best("draw_no_bet", lambda o: dnb(o, home))
    _q(rows, match, "Draw No Bet", f"Will {home} win (draw no bet)?", p, s)
    p, s = best("draw_no_bet", lambda o: dnb(o, away))
    _q(rows, match, "Draw No Bet", f"Will {away} win (draw no bet)?", p, s)

    # ---- 1st-half result (3-way) ----
    def h1(outs, sel):
        o = {x["name"]: x["price"] for x in outs}
        if home in o and away in o and "Draw" in o:
            dv = devig([o[home], o["Draw"], o[away]])
            idx = {"H": 0, "D": 1, "A": 2}[sel]
            return float(dv[idx]) if dv is not None else None
        return None
    for sel, lab in (("H", f"Will {home} lead at half-time?"),
                     ("D", "Will it be level at half-time?"),
                     ("A", f"Will {away} lead at half-time?")):
        p, s = best("h2h_h1", lambda o, sel=sel: h1(o, sel))
        _q(rows, match, "1st Half", lab, p, s)

    # ---- 1st-half totals (de-vig per available point) ----
    def h1_total(outs, pt):
        ov = price(outs, name="Over", point=pt)
        un = price(outs, name="Under", point=pt)
        return devig_two(ov, un) if ov and un else None
    for pt in (0.5, 1.0, 1.5):
        p, s = best("totals_h1", lambda o, pt=pt: h1_total(o, pt))
        _q(rows, match, "1st Half", f"Will there be over {pt} goals in the 1st half?", p, s)

    # ---- Team totals (Over/Under per team) ----
    def team_tot(outs, team, pt):
        ov = price(outs, name="Over", description=team, point=pt)
        un = price(outs, name="Under", description=team, point=pt)
        return devig_two(ov, un) if ov and un else None
    for team in (home, away):
        for pt in (0.5, 1.5):
            p, s = best("team_totals", lambda o, t=team, pt=pt: team_tot(o, t, pt))
            _q(rows, match, "Team Total", f"Will {team} score over {pt}?", p, s)

    # ---- Total corners (Over/Under per point) ----
    def corners(outs, pt):
        ov = price(outs, name="Over", point=pt)
        un = price(outs, name="Under", point=pt)
        return devig_two(ov, un) if ov and un else None
    for pt in (8.5, 9.5, 10.5):
        p, s = best("alternate_totals_corners", lambda o, pt=pt: corners(o, pt))
        _q(rows, match, "Corners", f"Will there be over {pt} total corners?", p, s)

    # ---- Total cards/bookings (Over/Under per point) ----
    for pt in (2.5, 3.5, 4.5):
        p, s = best("alternate_totals_cards", lambda o, pt=pt: corners(o, pt))
        _q(rows, match, "Cards", f"Will there be over {pt} total cards?", p, s)

    # ---- Player props (Yes-only -> margin removal, dedup by player pref. pinnacle) ----
    rows += _player_yes(mbb, match, "player_goal_scorer_anytime",
                        lambda pl: f"Will {pl} score a goal? (anytime)", "Anytime Scorer")
    rows += _player_yes(mbb, match, "player_to_receive_card",
                        lambda pl: f"Will {pl} be carded?", "Player Cards")
    rows += _player_shots(mbb, match)
    return rows


def _player_yes(mbb, match, key, qfn, mkt):
    """Yes-only player market: pick best (lowest-margin) book per player, strip margin."""
    best_per = {}  # player -> (prob, book)
    for bk in _ordered_books(mbb, key):
        factor = PROP_MARGIN.get(bk, PROP_MARGIN_DEFAULT)
        for o in mbb[bk][key]["outcomes"]:
            if o.get("name") != "Yes" or "description" not in o:
                continue
            pl = o["description"]
            p = clip_prob((1.0 / o["price"]) * factor)
            if pl not in best_per:  # first book in pref order wins
                best_per[pl] = (p, bk)
    rows = []
    for pl, (p, bk) in best_per.items():
        src = "sharp:pinnacle" if bk == "pinnacle" else f"book:{bk}"
        _q(rows, match, mkt, qfn(pl), p, src, note="Yes-only, margin-stripped")
    return rows


def _player_shots(mbb, match):
    """Player shots on target over 0.5 / 1.5 (Over-only -> margin removal)."""
    best_per = {}  # (player, point) -> (prob, book)
    for bk in _ordered_books(mbb, "player_shots_on_target"):
        factor = PROP_MARGIN.get(bk, PROP_MARGIN_DEFAULT)
        for o in mbb[bk]["player_shots_on_target"]["outcomes"]:
            if o.get("name") != "Over" or "description" not in o:
                continue
            pt = o.get("point")
            if pt not in (0.5, 1.5):
                continue
            k = (o["description"], pt)
            p = clip_prob((1.0 / o["price"]) * factor)
            if k not in best_per:
                best_per[k] = (p, bk)
    rows = []
    for (pl, pt), (p, bk) in best_per.items():
        src = "sharp:pinnacle" if bk == "pinnacle" else f"book:{bk}"
        _q(rows, match, "Shots on Target", f"Will {pl} have over {pt} shots on target?",
           p, src, note="Over-only, margin-stripped")
    return rows
