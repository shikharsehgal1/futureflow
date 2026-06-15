"""
simulate_wc.py — Monte-Carlo tournament simulation, aligned with Luke Benz's method.

Matches his deployed logic exactly:
  * goals ~ independent Poisson, lambda = exp(mu + alpha_att + delta_def + loc)  (lbenz_model)
  * group order: points -> goal diff -> goals scored  (head-to-head mini-table omitted;
    rarely binds and the simulated goals already break most ties)
  * 24 group top-2 + best-8 third-place advance (2026 48-team / 12-group format)
  * R32 bracket geometry + 3rd-place-slot assignment from Benz's helpers.R /
    data/third_place_combinations.csv (the official FIFA lookup)
  * knockout: extra time = +Poisson(lambda/3); penalty shootout = 50/50 coin flip
  * 10,000 sims -> P(advance / reach R16 / QF / SF / Final / win cup) per team

Output: data/wc_sim_results.csv
"""
from __future__ import annotations
import csv
import os
import numpy as np
from lbenz_model import load_ratings, get_rating, MU, NEUTRAL_FIELD, HOME_FIELD, HOSTS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
N_SIMS = 10000
RNG = np.random.default_rng(12345)

# R32 bracket geometry (from Benz helpers.R build_knockout_bracket), 0-indexed
T1 = [("E", 1), ("I", 1), ("A", 2), ("F", 1), ("K", 2), ("H", 1), ("D", 1), ("G", 1),
      ("C", 1), ("E", 2), ("A", 1), ("L", 1), ("J", 1), ("D", 2), ("B", 1), ("K", 1)]
T2 = [("3rd", "m74"), ("3rd", "m77"), ("B", 2), ("C", 2), ("L", 2), ("J", 2),
      ("3rd", "m81"), ("3rd", "m82"), ("F", 2), ("I", 2), ("3rd", "m79"), ("3rd", "m80"),
      ("H", 2), ("G", 2), ("3rd", "m85"), ("3rd", "m87")]


def load_groups():
    """group_letter -> list of teams, and list of (group, home, away) fixtures."""
    groups, fixtures = {}, []
    for r in csv.DictReader(open(os.path.join(DATA, "wc_schedule.csv"))):
        g = r.get("group", "").replace("Group ", "").strip()
        if not g or len(g) != 1:
            continue
        h, a = r["team1"].strip(), r["team2"].strip()
        groups.setdefault(g, set()).update([h, a])
        fixtures.append((g, h, a))
    return {g: sorted(t) for g, t in groups.items()}, fixtures


def load_tpc():
    tpc = {}
    p = os.path.join(DATA, "third_place_combinations.csv")
    for r in csv.DictReader(open(p)):
        tpc[r["groups"]] = {k: v for k, v in r.items() if k != "groups"}
    return tpc


def lam(ratings, a, b, host_a=False, host_b=False):
    ra, rb = get_rating(ratings, a), get_rating(ratings, b)
    if ra is None or rb is None:
        return 1.3, 1.3
    la = float(np.exp(MU + ra[0] + rb[1] + (HOME_FIELD if host_a else NEUTRAL_FIELD if not host_b else 0.0)))
    lb = float(np.exp(MU + rb[0] + ra[1] + (HOME_FIELD if host_b else NEUTRAL_FIELD if not host_a else 0.0)))
    return la, lb


def main():
    ratings = load_ratings()
    groups, fixtures = load_groups()
    tpc = load_tpc()
    teams = sorted({t for ts in groups.values() for t in ts})

    # group-stage fixture lambdas (team1 is home; host nation gets home_field)
    fx = []
    for g, h, a in fixtures:
        host_h, host_a = h in HOSTS, a in HOSTS
        lh, la = lam(ratings, h, a, host_a=host_h, host_b=host_a)
        fx.append((g, h, a, lh, la))
    # pre-draw all group goals: shape (n_fixtures, N_SIMS)
    gh = np.array([RNG.poisson(lh, N_SIMS) for _, _, _, lh, _ in fx])
    ga = np.array([RNG.poisson(la, N_SIMS) for _, _, _, _, la in fx])

    # neutral KO lambdas for every ordered pair (cached)
    kol = {}
    for x in teams:
        for y in teams:
            if x != y:
                kol[(x, y)] = lam(ratings, x, y)  # both neutral

    def ko(a, b):
        la, lb = kol[(a, b)]
        ga_, gb_ = RNG.poisson(la), RNG.poisson(lb)
        if ga_ == gb_:
            ga_ += RNG.poisson(la / 3); gb_ += RNG.poisson(lb / 3)
            if ga_ == gb_:
                return a if RNG.random() < 0.5 else b
        return a if ga_ > gb_ else b

    tally = {t: dict(adv=0, r16=0, qf=0, sf=0, final=0, champ=0) for t in teams}

    for s in range(N_SIMS):
        # ---- group standings ----
        st = {g: {t: [0, 0, 0] for t in ts} for g, ts in groups.items()}  # pts, gd, gf
        for i, (g, h, a, _, _) in enumerate(fx):
            hs, as_ = int(gh[i, s]), int(ga[i, s])
            st[g][h][1] += hs - as_; st[g][h][2] += hs
            st[g][a][1] += as_ - hs; st[g][a][2] += as_
            if hs > as_: st[g][h][0] += 3
            elif hs < as_: st[g][a][0] += 3
            else: st[g][h][0] += 1; st[g][a][0] += 1
        place = {}            # group -> ranked teams
        thirds = []
        for g, tbl in st.items():
            ranked = sorted(tbl, key=lambda t: (tbl[t][0], tbl[t][1], tbl[t][2]), reverse=True)
            place[g] = ranked
            for t in ranked[:2]:
                tally[t]["adv"] += 1
            thirds.append((g, ranked[2], st[g][ranked[2]]))
        # best 8 third-place
        thirds.sort(key=lambda x: (x[2][0], x[2][1], x[2][2]), reverse=True)
        qual3 = thirds[:8]
        for _, t, _ in qual3:
            tally[t]["adv"] += 1
        key = "".join(sorted(g for g, _, _ in qual3))
        combo = tpc.get(key)
        third_of = {g: place[g][2] for g, _, _ in qual3}

        def slot(spec):
            kind, v = spec          # kind=group letter & v=position, OR kind="3rd" & v=slot
            if kind == "3rd":
                if combo and combo.get(v) in place:
                    return place[combo[v]][2]
                return qual3[int(v[1:]) % len(qual3)][1]   # fallback
            return place[kind][v - 1]

        r32 = [(slot(T1[i]), slot(T2[i])) for i in range(16)]
        # ---- knockout ----
        w = [ko(a, b) for a, b in r32]                 # 16 R32 winners (reach R16)
        for t in w: tally[t]["r16"] += 1
        for _ in range(4):                              # R16->QF->SF->Final
            stage = {16: "qf", 8: "sf", 4: "final"}.get(len(w))
            w = [ko(w[2 * j], w[2 * j + 1]) for j in range(len(w) // 2)]
            if stage:
                for t in w: tally[t][stage] += 1
        tally[w[0]]["champ"] += 1

    rows = []
    for t in teams:
        d = tally[t]
        rows.append(dict(team=t,
                         p_advance=round(d["adv"] / N_SIMS, 4),
                         p_r16=round(d["r16"] / N_SIMS, 4),
                         p_qf=round(d["qf"] / N_SIMS, 4),
                         p_sf=round(d["sf"] / N_SIMS, 4),
                         p_final=round(d["final"] / N_SIMS, 4),
                         p_champ=round(d["champ"] / N_SIMS, 4)))
    rows.sort(key=lambda r: -r["p_champ"])
    with open(os.path.join(DATA, "wc_sim_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Simulated {N_SIMS} tournaments -> data/wc_sim_results.csv")
    print(f"\n{'team':22s} {'adv':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'CHAMP':>6}")
    for r in rows[:12]:
        print(f"{r['team']:22s} {100*r['p_advance']:5.0f}% {100*r['p_r16']:5.0f}% {100*r['p_qf']:5.0f}% "
              f"{100*r['p_sf']:5.0f}% {100*r['p_final']:5.0f}% {100*r['p_champ']:5.1f}%")


if __name__ == "__main__":
    main()
