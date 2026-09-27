# PromptShield

**A firewall for AI agents.** PromptShield scans emails, documents and web pages for hidden instructions before an AI agent reads them, and holds any risky action those instructions trigger before it runs.

Team **Goats** · ASYNC'26 · Track 3: Cybersecurity & Defense

![tests](https://github.com/<your-org>/promptshield/actions/workflows/tests.yml/badge.svg)

---

## The problem

AI agents now read our email, open documents and browse the web, then act on what they read. Attackers hide instructions inside that content (white-on-white text, HTML comments, invisible Unicode). The agent can't tell data from commands, so it obeys. This is **indirect prompt injection**. One poisoned email can make an agent forward private mail with no malware and no click from the victim.

## How PromptShield works: two checkpoints

```
                 ┌──────────── Checkpoint 1: scan content going in ────────────┐
 email / doc ──► │ clean-up + hidden-text extraction → 3 fast signals → score │ ──► agent (hidden text stripped)
                 │   rules · similarity to known attacks · injection classifier │      or QUARANTINE
                 │   LLM judge only for unclear cases                           │
                 └──────────────────────────────────────────────────────────────┘
                 ┌──────────── Checkpoint 2: hold risky actions ───────────────┐
 agent wants ──► │ policy (tool risk) + taint check: did this argument come     │ ──► run / ask a human / block
 to call a tool  │ from untrusted content the user never mentioned?             │      (everything audited)
                 └──────────────────────────────────────────────────────────────┘
```

Detection is probabilistic, so we don't rely on it alone. Even an attack no detector catches can't trigger a risky action without a human seeing where that instruction came from.

## Two-line integration

```python
from promptshield import Shield, ActionHeld

shield = Shield(policy="policies/email_agent.yaml")
shield.set_user_request("Summarise my inbox")            # what the USER actually asked

result = shield.scan(email_html, source="email")         # checkpoint 1
if result.level != "quarantine":
    agent_context.append(result.clean_text)              # hidden text already removed

@shield.guard_tool()                                     # checkpoint 2
def forward_email(email_id: str, to: str): ...
```

Other languages can use the HTTP API: `uvicorn api.main:app` → interactive docs at `http://localhost:8000/docs`.

## Quick start

```bash
git clone https://github.com/<your-org>/promptshield && cd promptshield
./setup.sh              # Windows: setup.bat     (add --ml for the embedding model + classifier)
source .venv/bin/activate
python -m demo_agent.run_demo          # before/after in the terminal
streamlit run dashboard/app.py         # dashboard
python scripts/preflight.py            # demo-day readiness check (--full loads models, asks Ollama)
```

Real-LLM demo: install [Ollama](https://ollama.com), `ollama pull qwen2.5:3b`, then `python -m demo_agent.run_demo --brain ollama`.

## Results

<!-- Paste eval/results.md here after running `python -m eval.run_eval` on your final data. -->
_Placeholder: numbers from the tiny starter seed set only. Replace with results on the full test set before submission._

## Repository layout

| path | what it is |
|---|---|
| `promptshield/` | the SDK: `preprocess.py`, `media.py`, `signals/`, `fuse.py`, `taint.py`, `gate.py`, `audit.py`, `shield.py` |
| `policies/email_agent.yaml` | which tools are risky and which arguments to taint-check |
| `demo_agent/` | a mock email agent (no real email is ever sent), with a real-LLM or simulated brain |
| `api/main.py` | FastAPI service |
| `dashboard/app.py` | Streamlit dashboard: live demo, approvals, try-to-break-it, audit log, evaluation |
| `data/`, `scripts/` | datasets, scripts to download / split them, and `preflight.py` (demo-day check) |
| `train/train_fusion.py` | learns how much to trust each signal |
| `eval/run_eval.py` | every number we report |
| `tests/` | pytest suite (runs in CI) |
| `docs/` | implementation guide, demo script, judge Q&A |

## Limitations (we know where the system stops)

- Only inline CSS is understood; text hidden via `<style>` classes or external stylesheets counts as visible.
- Image OCR and audio processing depend on optional system binaries (`tesseract` and `whisper`); missing binaries fall back gracefully to standard text and EXIF metadata extraction.
- Detection can be evaded by a determined attacker. The action gate reduces the damage, but an action with no destination argument (e.g. "summarise wrongly") can't be taint-checked.
- The taint check matches values (addresses, URLs, numbers) after undoing simple obfuscation. It can't follow data that the agent paraphrases.
- Human approval can suffer from alert fatigue; we only ask for high-risk actions and show where the instruction came from.
- Long-term memory poisoning and multi-turn attacks are out of scope.

## Responsible use

We only attack our own mock agent and mock outbox. All addresses use the reserved `.example` domain. No real mailbox is touched.

## Work done before vs during the event

- **Before the event (22–28 Sep, mentoring week):** everything up to the git tag `v0.9-pre-event`. The initial scaffold was drafted with AI coding assistance and then reviewed, tested and extended by the team (see `data/LICENSES.md`).
- **During the event (30 Sep – 1 Oct):** every commit after `v0.9-pre-event`.
  - Added multi-modal media processing (`media.py`) for PDF extraction, EXIF metadata smuggling and Tesseract image OCR.

## Disclosure

All datasets, models, APIs and libraries, with licences: [`data/LICENSES.md`](data/LICENSES.md).

## Team

Avi Sinha (lead) · add names and roles here

## Licence

MIT
