"""Demo-day readiness check. Run it on the laptop you will present from.

    python scripts/preflight.py            # quick checks (a few seconds)
    python scripts/preflight.py --full     # also load the ML models, ask Ollama, and run the tests

Prints one line per check:  OK  /  WARN (demo still works, but fix before judging)  /  FAIL (fix now).
Exit code is 1 if anything FAILs, so you can also use it in scripts.

Run it (1) after setup, (2) the night before the final, (3) 30 minutes before you present,
with the Wi-Fi you'll actually have. Everything it downloads is cached for offline use afterwards.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN, YELLOW, RED, DIM, END = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"
results: list[str] = []


def report(status: str, what: str, detail: str = "") -> None:
    colour = {"OK": GREEN, "WARN": YELLOW, "FAIL": RED}[status]
    results.append(status)
    print(f"{colour}{status:4s}{END}  {what}" + (f"  {DIM}{detail}{END}" if detail else ""))


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- checks
def check_python() -> None:
    v = sys.version_info
    report("OK" if v >= (3, 10) else "FAIL", f"Python {v.major}.{v.minor}", "3.10+ needed")


def check_core_packages() -> None:
    missing = []
    for mod in ("bs4", "numpy", "pydantic", "yaml", "requests", "sklearn", "fastapi", "uvicorn", "streamlit"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    report("FAIL" if missing else "OK", "Core packages",
           ("missing: " + ", ".join(missing) + "  -> run ./setup.sh") if missing else "")


def check_data() -> None:
    splits = ROOT / "data" / "splits"
    if not (splits / "test.jsonl").exists():
        report("FAIL", "Data splits", "run: python scripts/make_splits.py")
        return
    n = {name: len(rows(splits / f"{name}.jsonl")) for name in ("train", "val", "test")}
    detail = f"train {n['train']} / val {n['val']} / test {n['test']}"
    if n["test"] < 100:
        report("WARN", "Data splits", detail + "  -> test set too small for credible numbers (guide step 8.3)")
    else:
        report("OK", "Data splits", detail)
    blind = ROOT / "data" / "team_test_blind.jsonl"
    team = ROOT / "data" / "team_corpus.jsonl"
    report("OK" if blind.exists() and rows(blind) else "WARN", "Blind test set written by a teammate",
           "" if blind.exists() else "create data/team_test_blind.jsonl")
    report("OK" if team.exists() and len(rows(team)) >= 100 else "WARN", "Team corpus",
           f"{len(rows(team)) if team.exists() else 0} rows (aim for ~80 per member)")


def check_eval_results() -> None:
    f = ROOT / "eval" / "results.json"
    if not f.exists():
        report("WARN", "Evaluation results", "run: python -m eval.run_eval  (the dashboard's Evaluation tab needs it)")
        return
    res = json.loads(f.read_text())
    age_h = (datetime.now() - datetime.fromisoformat(res["generated_at"])).total_seconds() / 3600
    detail = f"{res['test_set']}, generated {age_h:.0f} h ago, config {res['config']}"
    small = "(16 rows)" in res["test_set"] or "(0 rows)" in res["test_set"]
    report("WARN" if small else "OK", "Evaluation results", detail + ("  -> starter numbers only" if small else ""))


def check_readme() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    todo = [p for p in ("<your-org>", "add names and roles here", "Placeholder: numbers") if p in text]
    report("WARN" if todo else "OK", "README filled in", ("still has: " + ", ".join(todo)) if todo else "")


def check_git() -> None:
    try:
        tags = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        report("WARN", "Git", "git not found")
        return
    if tags.returncode != 0:
        report("WARN", "Git", "not a git repository yet (guide Phase 0)")
        return
    has = "v0.9-pre-event" in tags.stdout.split()
    report("OK" if has else "WARN", "Tag v0.9-pre-event", "" if has else "create it before 28 Sep submission (step 8.6)")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report("WARN" if dirty else "OK", "Everything committed", f"{len(dirty.splitlines())} uncommitted file(s)" if dirty else "")


def check_runtime_db() -> None:
    try:
        from promptshield.audit import AuditLog
        AuditLog()
        report("OK", "Audit database writable", "runtime/promptshield.db")
    except Exception as exc:
        report("FAIL", "Audit database writable", str(exc)[:100])


def check_ollama(full: bool) -> None:
    import requests
    from promptshield.config import SETTINGS
    try:
        tags = requests.get(f"{SETTINGS.ollama_url}/api/tags", timeout=3).json()
    except Exception:
        report("WARN", "Ollama running", "start `ollama serve` (needed for the real-LLM demo and the judge)")
        return
    names = {m.get("name", "") for m in tags.get("models", [])}
    want = SETTINGS.ollama_model
    pulled = want in names or f"{want}:latest" in names
    report("OK" if pulled else "FAIL", f"Ollama model {want}", "" if pulled else f"run: ollama pull {want}")
    if not (full and pulled):
        return
    # Does the real model actually fall for the attack? (guide step 8.1)
    from demo_agent.agent import EmailAgent
    from demo_agent.brains import OllamaBrain
    from demo_agent.mailstore import MailStore
    from demo_agent.run_demo import TASK
    t0 = time.perf_counter()
    leaked = []
    for eid in ("e3", "e7", "e9"):
        run = EmailAgent(MailStore.load(only_ids=[eid]), OllamaBrain()).run(TASK)
        if run.leaked:
            leaked.append(eid)
    secs = time.perf_counter() - t0
    report("OK" if leaked else "WARN", "Real LLM falls for the attack (unshielded)",
           f"leaked on {leaked or 'none'} in {secs:.0f}s" + ("" if leaked else
           "  -> try another model or use the simulated brain and say so (step 8.1)"))


def check_ml(full: bool) -> None:
    try:
        importlib.import_module("sentence_transformers")
        importlib.import_module("transformers")
    except ImportError:
        report("WARN", "ML signals installed", "optional: ./setup.sh --ml (embeddings + classifier)")
        return
    if not full:
        report("OK", "ML packages installed", "use --full to load the models")
        return
    from promptshield.signals import classifier, similarity
    t0 = time.perf_counter()
    backend = similarity.backend_name()
    report("OK" if backend == "embeddings" else "WARN", "Similarity backend", f"{backend} ({time.perf_counter()-t0:.1f}s)")
    t0 = time.perf_counter()
    ok = classifier.is_available()
    report("OK" if ok else "WARN", "Injection classifier loads", f"{time.perf_counter()-t0:.1f}s" if ok else
           "check PS_CLASSIFIER_MODEL / huggingface-cli login")


def check_scan_speed() -> None:
    from promptshield import Shield
    from demo_agent.mailstore import MailStore
    s = Shield(db_path=ROOT / "runtime" / "preflight.db", use_judge=False)
    email = MailStore.load(only_ids=["e3"]).inbox[0]
    s.scan("warm-up")
    r = s.scan(email["body"], source="preflight")
    ok = r.level == "quarantine"
    report("OK" if ok else "FAIL", "Demo attack e3 is quarantined", f"risk {r.risk}, {r.latency_ms} ms")
    try:
        (ROOT / "runtime" / "preflight.db").unlink(missing_ok=True)
    except PermissionError:
        pass


def check_tests() -> None:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    last = (p.stdout.strip().splitlines() or ["no output"])[-1]
    report("OK" if p.returncode == 0 else "FAIL", "Test suite", last)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="load models, query Ollama, run pytest")
    args = ap.parse_args()

    print(f"PromptShield preflight  ({'full' if args.full else 'quick'})\n")
    check_python()
    check_core_packages()
    if "FAIL" in results:
        return 1
    check_runtime_db()
    check_scan_speed()
    check_data()
    check_eval_results()
    check_ml(args.full)
    check_ollama(args.full)
    check_readme()
    check_git()
    if args.full:
        check_tests()

    fails, warns = results.count("FAIL"), results.count("WARN")
    print(f"\n{RED if fails else YELLOW if warns else GREEN}{fails} fail, {warns} warning(s){END}")
    if not fails:
        print("Demo will run. Clear the warnings before judging where you can.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
