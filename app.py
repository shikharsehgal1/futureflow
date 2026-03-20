"""
Football Prediction Dashboard — Flask App
Run: python3.10 app.py
"""

import csv
import io
import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# ── Job tracking for SSE streaming ───────────────────────────────────────────
jobs = {}  # job_id -> {"lines": [...], "done": bool, "returncode": int|None}
job_lock = threading.Lock()
running_lock = threading.Lock()
running_process = False  # prevent concurrent pipeline runs
running_since = None  # timestamp when process started
MAX_RUNNING_TIME = 600  # auto-unlock after 10 minutes (safety net)

# ── Allowed scripts (whitelist for security) ─────────────────────────────────
SCRIPTS = {
    "fetch_fixtures":       ["python3.10", "preliminarymodel/fetch_fixtures.py"],
    "fetch_fixtures_results": ["python3.10", "preliminarymodel/fetch_fixtures.py", "--results"],
    "update_data":          ["python3.10", "preliminarymodel/update_data.py"],
    "tm_injuries":          ["python3.10", "tm_injuries.py"],
    "tm_injuries_all":      ["python3.10", "tm_injuries.py"],
    "ratings":              ["python3.10", "mainmodel/part7_ratings.py"],
    "predict":              ["python3.10", "mainmodel/part7_predict.py"],
    "build_cross_ratings":  ["python3.10", "crossleague/build_cross_ratings.py"],
    "predict_cl":           ["python3.10", "crossleague/predict_cl.py"],
    "tm_valuations":        ["python3.10", "tm_valuations.py"],
}

# ── Pipeline definitions ─────────────────────────────────────────────────────
PIPELINES = {
    "friday": [
        ("Fetch Fixtures & Weather", "fetch_fixtures"),
        ("Scrape Injuries", "tm_injuries"),
        ("Refit Ratings", "ratings"),
        ("Generate Predictions", "predict"),
    ],
    "monday": [
        ("Fetch New Results", "update_data"),
        ("Update Fixtures", "fetch_fixtures_results"),
    ],
    "european": [
        ("Rebuild Cross Ratings", "build_cross_ratings"),
        ("Predict CL/Europa", "predict_cl"),
    ],
}


def read_csv_as_dicts(path):
    """Read a CSV file and return list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def file_mod_time(path):
    """Return last modified time as ISO string, or None."""
    if os.path.exists(path):
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    return None


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    rows = read_csv_as_dicts(DATA / "predictions.csv")
    # Merge injury notes if available
    injuries = {}
    inj_rows = read_csv_as_dicts(DATA / "tm_injuries.csv")
    for r in inj_rows:
        key = (r.get("Date", ""), r.get("HomeTeam", ""), r.get("AwayTeam", ""))
        injuries[key] = r
    for row in rows:
        key = (row.get("Date", ""), row.get("HomeTeam", ""), row.get("AwayTeam", ""))
        inj = injuries.get(key, {})
        row["injury_adj_net"] = inj.get("injury_adj_net", "")
        row["injury_notes"] = inj.get("injury_notes", "")
    return jsonify(rows)


@app.route("/api/ratings")
def api_ratings():
    path = DATA / "ratings.json"
    if not os.path.exists(path):
        return jsonify({"error": "ratings.json not found — run part7_ratings.py first"}), 404
    with open(path, "r") as f:
        return jsonify(json.load(f))


@app.route("/api/injuries")
def api_injuries():
    return jsonify(read_csv_as_dicts(DATA / "tm_injuries.csv"))


@app.route("/api/cross-ratings")
def api_cross_ratings():
    path = BASE / "crossleague" / "data" / "cross_ratings.csv"
    return jsonify(read_csv_as_dicts(path))


@app.route("/api/history")
def api_history():
    return jsonify(read_csv_as_dicts(DATA / "predictions_log.csv"))


@app.route("/api/status")
def api_status():
    files = {
        "fixtures.csv": str(DATA / "fixtures.csv"),
        "tm_injuries.csv": str(DATA / "tm_injuries.csv"),
        "predictions.csv": str(DATA / "predictions.csv"),
        "predictions_log.csv": str(DATA / "predictions_log.csv"),
        "big5_with_probs.csv": str(DATA / "big5_with_probs.csv"),
        "ratings.json": str(DATA / "ratings.json"),
        "cross_ratings.csv": str(BASE / "crossleague" / "data" / "cross_ratings.csv"),
    }
    return jsonify({name: file_mod_time(path) for name, path in files.items()})


# ── Script execution with SSE ────────────────────────────────────────────────
def run_script_in_thread(job_id, cmd):
    """Run a subprocess, capturing output line by line into jobs[job_id]."""
    global running_process
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE),
        )
        for line in proc.stdout:
            with job_lock:
                jobs[job_id]["lines"].append(line.rstrip("\n"))
        proc.wait()
        with job_lock:
            jobs[job_id]["returncode"] = proc.returncode
            jobs[job_id]["done"] = True
    except Exception as e:
        with job_lock:
            jobs[job_id]["lines"].append(f"ERROR: {e}")
            jobs[job_id]["returncode"] = 1
            jobs[job_id]["done"] = True
    finally:
        with running_lock:
            running_process = False
            running_since = None


def check_stale_lock():
    """Auto-clear lock if stuck for too long."""
    global running_process, running_since
    with running_lock:
        if running_process and running_since and (time.time() - running_since > MAX_RUNNING_TIME):
            running_process = False
            running_since = None


@app.route("/api/run/<script>", methods=["POST"])
def api_run(script):
    global running_process, running_since
    if script not in SCRIPTS:
        return jsonify({"error": f"Unknown script: {script}"}), 400

    check_stale_lock()
    with running_lock:
        if running_process:
            return jsonify({"error": "A process is already running"}), 409
        running_process = True
        running_since = time.time()

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"lines": [], "done": False, "returncode": None, "script": script}

    cmd = SCRIPTS[script]
    t = threading.Thread(target=run_script_in_thread, args=(job_id, cmd), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/run-pipeline/<pipeline>", methods=["POST"])
def api_run_pipeline(pipeline):
    global running_process, running_since
    if pipeline not in PIPELINES:
        return jsonify({"error": f"Unknown pipeline: {pipeline}"}), 400

    check_stale_lock()
    with running_lock:
        if running_process:
            return jsonify({"error": "A process is already running"}), 409
        running_process = True
        running_since = time.time()

    job_id = str(uuid.uuid4())[:8]
    steps = PIPELINES[pipeline]
    with job_lock:
        jobs[job_id] = {
            "lines": [],
            "done": False,
            "returncode": None,
            "pipeline": pipeline,
            "steps": [{"name": name, "status": "pending"} for name, _ in steps],
        }

    def run_pipeline_thread():
        global running_process
        try:
            for i, (name, script_key) in enumerate(steps):
                with job_lock:
                    jobs[job_id]["steps"][i]["status"] = "running"
                    jobs[job_id]["lines"].append(f"\n▶ Step {i+1}/{len(steps)}: {name}")

                cmd = SCRIPTS[script_key]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(BASE),
                )
                for line in proc.stdout:
                    with job_lock:
                        jobs[job_id]["lines"].append(line.rstrip("\n"))
                proc.wait()

                with job_lock:
                    if proc.returncode == 0:
                        jobs[job_id]["steps"][i]["status"] = "done"
                        jobs[job_id]["lines"].append(f"✓ {name} completed")
                    else:
                        jobs[job_id]["steps"][i]["status"] = "error"
                        jobs[job_id]["lines"].append(f"✗ {name} failed (exit code {proc.returncode})")
                        jobs[job_id]["returncode"] = proc.returncode
                        jobs[job_id]["done"] = True
                        return

            with job_lock:
                jobs[job_id]["returncode"] = 0
                jobs[job_id]["done"] = True
                jobs[job_id]["lines"].append(f"\n✓ Pipeline complete!")
        except Exception as e:
            with job_lock:
                jobs[job_id]["lines"].append(f"ERROR: {e}")
                jobs[job_id]["returncode"] = 1
                jobs[job_id]["done"] = True
        finally:
            with running_lock:
                running_process = False

    t = threading.Thread(target=run_pipeline_thread, daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    def generate():
        sent = 0
        while True:
            with job_lock:
                job = jobs.get(job_id)
                if job is None:
                    yield f"data: {json.dumps({'error': 'Unknown job'})}\n\n"
                    return
                new_lines = job["lines"][sent:]
                done = job["done"]
                returncode = job["returncode"]
                steps = job.get("steps")

            for line in new_lines:
                msg = {"type": "output", "line": line}
                yield f"data: {json.dumps(msg)}\n\n"
                sent += 1

            if done:
                msg = {"type": "done", "returncode": returncode}
                if steps:
                    msg["steps"] = steps
                yield f"data: {json.dumps(msg)}\n\n"
                return

            time.sleep(0.3)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/predict-cl", methods=["POST"])
def api_predict_cl():
    """Run predict_cl.py with custom fixture passed via temp CSV."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Write temp fixtures CSV
    tmp_path = BASE / "crossleague" / "data" / "_tmp_fixture.csv"
    with open(tmp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "HomeTeam", "AwayTeam", "Competition", "OddsH", "OddsD", "OddsA"])
        w.writerow([
            data.get("date", ""),
            data.get("home", ""),
            data.get("away", ""),
            data.get("competition", ""),
            data.get("oddsH", ""),
            data.get("oddsD", ""),
            data.get("oddsA", ""),
        ])

    try:
        result = subprocess.run(
            ["python3.10", "crossleague/predict_cl.py", "--fixtures", str(tmp_path)],
            capture_output=True, text=True, cwd=str(BASE), timeout=60,
        )
        return jsonify({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Prediction timed out"}), 504
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
