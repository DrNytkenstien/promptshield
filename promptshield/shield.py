"""The public API developers use: the Shield class.

    shield = Shield(policy="policies/email_agent.yaml")

    # Checkpoint 1: scan untrusted content before the agent sees it
    result = shield.scan(email_html, source="email")
    if result.level != "quarantine":
        agent_context.append(result.clean_text)      # hidden text is already stripped out

    # Checkpoint 2: wrap every tool
    @shield.guard_tool()
    def forward_email(email_id: str, to: str): ...

This file wires together: preprocess -> signals -> fuse -> taint -> gate -> audit.
"""
from __future__ import annotations

import functools
import inspect
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import fuse
from .audit import AuditLog
from .config import POLICY_DIR
from .gate import ActionGate, Policy
from .media import ingest_media
from .preprocess import preprocess, qr_rules_score
from .signals import classifier, heuristics, judge, similarity
from .taint import SessionTaint, extract_values
from .types import Finding, GateDecision, ScanResult, SignalScore


def _rules_score(chunk) -> SignalScore:
    """The 'heuristics' (rules) feature for one chunk: text rules plus the continuous QR/media rule score.

    Same signal name as before, so the fusion feature schema is unchanged. Nothing here decides
    allow/warn/quarantine; fuse.py does that from the signal values.
    """
    base = heuristics.score_chunk(chunk)
    qr, reasons = qr_rules_score(chunk.text)
    if qr > base.score:
        return SignalScore(name="heuristics", score=qr, reasons=list(base.reasons) + reasons)
    return base


class ActionHeld(Exception):
    """Raised by a guarded tool when the gate holds or blocks the call."""

    def __init__(self, decision: GateDecision) -> None:
        self.decision = decision
        verb = "blocked" if decision.decision == "block" else "held for human approval"
        super().__init__(f"{decision.tool} {verb}: " + "; ".join(decision.reasons))


class Shield:
    def __init__(self, policy: Path | str = POLICY_DIR / "email_agent.yaml",
                 db_path: Optional[Path | str] = None, use_judge: bool = True,
                 use_similarity: bool = True, use_classifier: bool = True,
                 use_heuristics: bool = True) -> None:
        self.policy = Policy(policy)
        self.gate = ActionGate(self.policy)
        self.audit = AuditLog(db_path)
        self.use_judge = use_judge
        self.use = {"heuristics": use_heuristics, "similarity": use_similarity, "classifier": use_classifier}
        self._sessions: dict[str, SessionTaint] = {}
        self._tools: dict[str, Callable] = {}
        self.created_at = time.time()          # lets a UI show only actions from this run

    # ================================================================ sessions
    def session(self, session_id: str = "default") -> SessionTaint:
        if session_id not in self._sessions:
            s = SessionTaint()
            s.add_trusted(self.policy.trusted_values)
            self._sessions[session_id] = s
        return self._sessions[session_id]

    def set_user_request(self, text: str, session_id: str = "default") -> None:
        """Tell the gate what the USER actually asked for. Values in it are trusted."""
        self.session(session_id).add_trusted(text)

    def reset_session(self, session_id: str = "default") -> None:
        self._sessions.pop(session_id, None)

    # ================================================================ checkpoint 1
    def scan(self, content: Any, source: str = "content", session_id: str = "default",
             label: str = "") -> ScanResult:
        t0 = time.perf_counter()

        # Media Normalization: Extract text signals from Images, PDFs, and Audio
        if source in ("image", "pdf", "audio") or isinstance(content, (Path, bytes, bytearray)):
            media_type = source if source in ("image", "pdf", "audio") else "auto"
            content, source = ingest_media(content, media_type=media_type)

        pre = preprocess(content)
        chunks = pre.chunks

        per_chunk: list[list] = [[] for _ in chunks]
        if chunks:
            if self.use["heuristics"]:
                for i, c in enumerate(chunks):
                    per_chunk[i].append(_rules_score(c))
            if self.use["similarity"]:
                for i, s in enumerate(similarity.score_chunks(chunks)):
                    per_chunk[i].append(s)
            if self.use["classifier"]:
                for i, s in enumerate(classifier.score_chunks(chunks)):
                    per_chunk[i].append(s)

        judge_fn = judge.judge_chunk if self.use_judge else None
        results = [fuse.combine_chunk(c, per_chunk[i], judge_fn) for i, c in enumerate(chunks)]
        risk = fuse.document_risk(results, pre.findings)
        level = fuse.level_for(risk)
        top = max(results, key=lambda r: r.combined, default=None)
        explanation = fuse.explain(level, risk, top, pre.findings)

        # Everything (visible AND hidden) is tainted: attacker values often live in hidden text.
        all_text = pre.visible_text + "\n" + "\n".join(pre.hidden_texts)
        taint_id = self.session(session_id).add_untrusted(all_text, source, level, risk)

        summary: dict[str, Optional[float]] = {}
        for name in ("heuristics", "similarity", "classifier", "judge"):
            scores = [s.score for r in results for s in r.signals if s.name == name and s.available]
            summary[name] = max(scores) if scores else None     # None = didn't run / unavailable

        result = ScanResult(
            taint_id=taint_id, source=source, level=level, risk=risk,
            clean_text=pre.visible_text, explanation=explanation, findings=pre.findings,
            chunks=results, top_chunk=top, signal_summary=summary,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        self.audit.log_scan(session_id, result, label=label)
        return result

    # ================================================================ checkpoint 2
    def check_action(self, tool: str, args: dict[str, Any], session_id: str = "default") -> GateDecision:
        decision = self.gate.decide(tool, args, self.session(session_id))
        status = {"allow": "allowed", "require_approval": "pending", "block": "blocked"}[decision.decision]
        decision.action_id = self.audit.log_action(session_id, decision, status)
        return decision

    def guard_tool(self, name: Optional[str] = None, session_id: str = "default"):
        """Decorator: every call to the tool goes through the gate first."""
        def wrap(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            sig = inspect.signature(fn)
            self._tools[tool_name] = fn

            @functools.wraps(fn)
            def guarded(*a, **kw):
                sid = kw.pop("_session_id", session_id)
                bound = sig.bind(*a, **kw)
                bound.apply_defaults()
                decision = self.check_action(tool_name, dict(bound.arguments), sid)
                if decision.decision != "allow":
                    raise ActionHeld(decision)
                out = fn(*bound.args, **bound.kwargs)
                self.audit.set_status(decision.action_id, "executed", by="auto", result=out)
                return out

            guarded.__promptshield_tool__ = tool_name
            return guarded
        return wrap

    # ================================================================ human decisions
    def approve(self, action_id: int, by: str = "human", execute: bool = True) -> Any:
        row = self.audit.get_action(action_id)
        if row is None or row["status"] != "pending":
            raise ValueError(f"Action {action_id} is not pending")
        self.audit.set_status(action_id, "approved", by=by)
        if execute and row["tool"] in self._tools:
            import json
            out = self._tools[row["tool"]](**json.loads(row["args_json"]))
            self.audit.set_status(action_id, "executed", result=out)
            return out
        return None

    def deny(self, action_id: int, by: str = "human") -> None:
        self.audit.set_status(action_id, "denied", by=by)

    # ================================================================ output check
    def check_output(self, text: str, session_id: str = "default") -> list[Finding]:
        """Scan the agent's own reply for data-leaking links (e.g. markdown images with query strings)."""
        findings = [f for f in preprocess(text).findings if f.kind == "exfil_image"]
        sess = self.session(session_id)
        for v in extract_values(text):
            rec = sess.origin_of(v)
            if rec is not None and v.startswith("http"):
                findings.append(Finding(kind="tainted_link", detail=f"Reply contains a link from untrusted {rec.source}",
                                        text=v))
        return findings