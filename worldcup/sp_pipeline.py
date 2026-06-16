"""
sp_pipeline.py — One full SportsPredict cycle: pull live questions, solve, submit.

Run on a schedule so submissions stay refreshed with the latest model numbers right
up to each match's kickoff (markets lock at kickoff). Submits only OPEN markets, so
already-locked matches are skipped automatically.

  fetch_sp.py   -> data/sp_questions.csv   (live questions)
  predict_wc.py + apply_adjustments.py     (refresh model, optional)
  tilt_matches.py -> tilts wc_match_summary lambdas for selected matches
                     (clean baseline kept in wc_match_summary.csv.bak)
  sp_solver.py  -> data/sp_entries.csv     (map questions -> model probs)
  sp_submit.py                              (POST to open markets, with backoff)

Usage:  SP_API_KEY=... python3 sp_pipeline.py [--refresh-model]
"""
import os, subprocess, sys, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*cmd):
    print(f"  $ {' '.join(c.split('/')[-1] for c in cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=HERE).returncode


def main():
    if not os.environ.get("SP_API_KEY"):
        sys.exit("set SP_API_KEY")
    print(f"[sp_pipeline {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}]")
    run("python3", f"{HERE}/fetch_sp.py")
    if "--refresh-model" in sys.argv:
        run("python3", f"{HERE}/predict_wc.py")
        run("python3", f"{HERE}/apply_adjustments.py")
        # capture the freshly-regenerated honest numbers as the new clean baseline,
        # then re-apply the directional tilts on top.
        run("python3", f"{HERE}/tilt_matches.py", "--rebaseline")
    else:
        run("python3", f"{HERE}/tilt_matches.py")
    run("python3", f"{HERE}/sp_solver.py")
    run("python3", f"{HERE}/sp_submit.py", "--delay", "1.6")
    print("[sp_pipeline done]")


if __name__ == "__main__":
    main()
