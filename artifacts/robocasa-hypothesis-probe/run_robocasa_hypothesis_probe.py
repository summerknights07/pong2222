#!/usr/bin/env python3
"""Probe Adapt-1 hypothesis surfaces with diverse RoboCasa atomic-task samples.

The script never persists the API key. Pass it through ADAPT1_API_KEY.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = os.environ.get(
    "ADAPT1_BASE_URL", "https://rei-neuroadapt-api-uat.reilabs.org"
).rstrip("/")
API_KEY = os.environ["ADAPT1_API_KEY"]
RUN_TAG = f"robocasa-hyp-{int(time.time())}-{random.randrange(1000, 9999)}"
OUT_DIR = Path(__file__).resolve().parent / "robocasa_hypothesis_probe" / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {0, 1, 2, 3, 5, 8, 12}

# These samples use task names, public task semantics, and success thresholds from
# RoboCasa atomic task definitions. They are structured API samples, not MuJoCo
# rollouts. Each event describes one observable before/action/after transition.
SAMPLES = [
    dict(task="OpenDrawer", fixture="drawer", action="pull", before=0.08, after=0.98, threshold=0.95, reached=True),
    dict(task="OpenDrawer", fixture="drawer", action="pull", before=0.10, after=0.62, threshold=0.95, reached=False),
    dict(task="CloseDrawer", fixture="drawer", action="push", before=0.82, after=0.03, threshold=0.05, reached=True),
    dict(task="CloseDrawer", fixture="drawer", action="push", before=0.76, after=0.21, threshold=0.05, reached=False),
    dict(task="SlideDishwasherRackOut", fixture="dishwasher_rack", action="pull", before=0.50, after=0.97, threshold=0.95, reached=True),
    dict(task="SlideDishwasherRackOut", fixture="dishwasher_rack", action="pull", before=0.48, after=0.73, threshold=0.95, reached=False),
    dict(task="SlideDishwasherRackIn", fixture="dishwasher_rack", action="push", before=0.55, after=0.04, threshold=0.05, reached=True),
    dict(task="SlideDishwasherRackIn", fixture="dishwasher_rack", action="push", before=0.58, after=0.19, threshold=0.05, reached=False),
    dict(task="OpenFridgeDrawer", fixture="fridge_drawer", action="pull", before=0.18, after=0.81, threshold=0.70, reached=True),
    dict(task="OpenFridgeDrawer", fixture="fridge_drawer", action="pull", before=0.20, after=0.53, threshold=0.70, reached=False),
    dict(task="CloseFridgeDrawer", fixture="fridge_drawer", action="push", before=0.90, after=0.03, threshold=0.05, reached=True),
    dict(task="CloseFridgeDrawer", fixture="fridge_drawer", action="push", before=0.88, after=0.27, threshold=0.05, reached=False),
]


def request(method: str, path: str, session_id: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "User-Agent": "curl/8.10.1",
        "X-Neuroadapt-Session-Id": session_id,
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    last = None
    for attempt in range(6):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
                latency = time.perf_counter() - started
                return response.status, json.loads(body) if body else {}, latency
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            latency = time.perf_counter() - started
            if exc.code not in {429, 502, 503, 504}:
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = {"raw": body}
                return exc.code, parsed, latency
            last = RuntimeError(f"HTTP {exc.code}: {body[:500]}")
        except Exception as exc:  # transient network path
            last = exc
        time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"request failed after retries: {method} {path}: {last}")


def log(condition: str, step: int, operation: str, status: int, latency: float, response: dict):
    row = {
        "condition": condition,
        "step": step,
        "operation": operation,
        "status": status,
        "latency_seconds": round(latency, 6),
        "response": response,
    }
    with (OUT_DIR / "trace.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def call(condition: str, step: int, operation: str, method: str, path: str, session: str, payload=None):
    status, response, latency = request(method, path, session, payload)
    log(condition, step, operation, status, latency, response)
    print(f"{condition:24s} step={step:02d} {operation:18s} HTTP={status} {latency:7.3f}s", flush=True)
    if not 200 <= status < 300:
        raise RuntimeError(f"{condition} {operation} failed HTTP {status}: {response}")
    return response


def sample_text(sample: dict, index: int) -> str:
    outcome = "TARGET_STATE_REACHED" if sample["reached"] else "TARGET_STATE_NOT_REACHED"
    return (
        f"ROBOCASA_ATOMIC_TRANSITION; SAMPLE_{index:02d}; TARGET_PRESENT; "
        f"MANIPULATION_EXECUTED; TASK_{sample['task'].upper()}; "
        f"FIXTURE_{sample['fixture'].upper()}; ACTION_{sample['action'].upper()}; {outcome}"
    )


def memory_checkpoint(condition: str, step: int, session: str):
    payload = {
        "session_id": session,
        "user_message": "What observable action-effect hypotheses are supported by these RoboCasa transitions?",
        "top_k": 30,
        "include_reasoning": True,
        "update_memory_state": False,
        "metadata_filter": {"run_id": RUN_TAG, "condition": condition},
    }
    return call(condition, step, "memory_explain", "POST", "/api/v1/memory/explain", session, payload)


def run_no_domain():
    condition = "no_domain_memory"
    session = f"{RUN_TAG}-nodomain"
    memory_checkpoint(condition, 0, session)
    for i, sample in enumerate(SAMPLES, 1):
        text = sample_text(sample, i)
        payload = {
            "session_id": session,
            "user_message": text,
            "ai_message": "",
            "context": {
                "run_id": RUN_TAG,
                "condition": condition,
                "sample_index": i,
                "task": sample["task"],
                "fixture": sample["fixture"],
                "action": sample["action"],
                "before_position": sample["before"],
                "after_position": sample["after"],
                "goal_reached": sample["reached"],
            },
        }
        call(condition, i, "memory_store", "POST", "/api/v1/memory/store", session, payload)
        if i in CHECKPOINTS:
            memory_checkpoint(condition, i, session)


def domain_definition(domain_id: str, session: str, with_hypotheses: bool):
    payload = {
        "domain_id": domain_id,
        "session_id": session,
        "description": "RoboCasa atomic fixture-manipulation transitions; public observable task interface only.",
        "schema": {
            "entities": ["robot", "fixture", "task", "transition"],
            "relations": ["robocasa_action_effect"],
            "signals": ["before_position", "after_position", "action_token", "outcome_token"],
            "event_types": ["robocasa_atomic_transition"],
        },
        "learning": {
            "enabled": True,
            "transition": {
                "enabled": True,
                "event_types": ["robocasa_atomic_transition"],
                "input_paths": ["values.before_position"],
                "action_path": "values.action_token",
                "targets": [{"path": "values.outcome_token", "type": "categorical"}],
                "group_by_paths": [],
                "required_support": 2,
                "max_samples": 128,
                "neighbors": 8,
                "max_distance": 0.75,
            },
        },
    }
    if with_hypotheses:
        payload["hypotheses"] = [
            {
                "name": "manipulation_reaches_target_state",
                "when": ["TARGET_PRESENT", "MANIPULATION_EXECUTED"],
                "predicts": ["TARGET_STATE_REACHED"],
                "falsified_by": ["TARGET_STATE_NOT_REACHED"],
                "weight": 1.0,
                "relation": "robocasa_action_effect",
                "policy": "predict_target_state_reached",
            },
            {
                "name": "manipulation_does_not_reach_target_state",
                "when": ["TARGET_PRESENT", "MANIPULATION_EXECUTED"],
                "predicts": ["TARGET_STATE_NOT_REACHED"],
                "falsified_by": ["TARGET_STATE_REACHED"],
                "weight": 1.0,
                "relation": "robocasa_action_effect",
                "policy": "predict_target_state_not_reached",
            },
        ]
    return payload


def domain_checkpoint(condition: str, step: int, session: str, domain_id: str):
    base = {
        "session_id": session,
        "question": "Given the observed RoboCasa fixture transitions, does manipulation reach the declared target state?",
        "top_k": 30,
        "metadata_filter": {"run_id": RUN_TAG, "condition": condition},
        "update_memory_state": False,
        "relation": "robocasa_action_effect",
        "context": {
            "values": {
                "target_state": "TARGET_PRESENT",
                "execution_state": "MANIPULATION_EXECUTED",
                "before_position": 0.5,
                "action_token": "pull",
            },
            "metadata": {"run_id": RUN_TAG, "condition": condition},
        },
        "allow_exploration": False,
    }
    query = dict(base)
    query["return_fields"] = [
        "ranked_hypotheses", "transition_prediction", "learning_state", "missing_evidence",
        "predicted_observations", "core_support", "supporting_memories", "selection"
    ]
    call(condition, step, "domain_query", "POST", f"/api/v1/domains/{domain_id}/query", session, query)
    call(condition, step, "domain_explain", "POST", f"/api/v1/domains/{domain_id}/explain", session, base)


def run_domain(with_hypotheses: bool, feedback_driven: bool = False):
    if feedback_driven:
        condition = "domain_feedback_candidates"
    else:
        condition = "domain_with_candidates" if with_hypotheses else "domain_no_candidates"
    session = f"{RUN_TAG}-{'hyp' if with_hypotheses else 'plain'}"
    domain_id = f"{RUN_TAG}-{'hyp-domain' if with_hypotheses else 'plain-domain'}"
    call(condition, 0, "create_domain", "POST", "/api/v1/domains", session, domain_definition(domain_id, session, with_hypotheses))
    domain_checkpoint(condition, 0, session, domain_id)
    for i, sample in enumerate(SAMPLES, 1):
        outcome = "TARGET_STATE_REACHED" if sample["reached"] else "TARGET_STATE_NOT_REACHED"
        payload = {
            "session_id": session,
            "event_type": "robocasa_atomic_transition",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entities": {
                "robot": "panda_omron",
                "fixture": sample["fixture"],
                "task": sample["task"],
                "transition": f"sample_{i:02d}",
            },
            "values": {
                "before_position": sample["before"],
                "after_position": sample["after"],
                "action_token": sample["action"],
                "outcome_token": outcome,
                "goal_reached": sample["reached"],
                "success_threshold": sample["threshold"],
            },
            "relations": [{
                "subject": sample["action"],
                "relation": "robocasa_action_effect",
                "object": outcome,
                "observed": True,
            }],
            "text": sample_text(sample, i),
            "metadata": {
                "run_id": RUN_TAG,
                "condition": condition,
                "sample_index": i,
                "source": "official_robocasa_atomic_task_semantics",
                "contains_hidden_state": False,
                "contains_evaluative_feedback": False,
            },
        }
        call(condition, i, "domain_event", "POST", f"/api/v1/domains/{domain_id}/events", session, payload)
        if feedback_driven:
            policy_outcomes = {
                "predict_target_state_reached": sample["reached"],
                "predict_target_state_not_reached": not sample["reached"],
            }
            for policy, correct in policy_outcomes.items():
                feedback = {
                    "session_id": session,
                    "outcome": "positive" if correct else "negative",
                    "values": {"reward": 1.0 if correct else 0.0},
                    "text": f"Observed {outcome}; {policy} was {'supported' if correct else 'counterevidenced'}.",
                    "metadata": {
                        "run_id": RUN_TAG,
                        "condition": condition,
                        "sample_index": i,
                        "relation": "robocasa_action_effect",
                        "policy": policy,
                    },
                    "feedback_kind": "execution",
                    "relation": "robocasa_action_effect",
                    "policy": policy,
                }
                call(condition, i, f"feedback_{policy}", "POST", f"/api/v1/domains/{domain_id}/feedback", session, feedback)
        if i in CHECKPOINTS:
            domain_checkpoint(condition, i, session, domain_id)


def main():
    mode = os.environ.get("PROBE_MODE", "all")
    manifest = {
        "run_tag": RUN_TAG,
        "base_url": BASE_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count_per_condition": len(SAMPLES),
        "checkpoints": sorted(CHECKPOINTS),
        "conditions": ["no_domain_memory", "domain_no_candidates", "domain_with_candidates"],
        "probe_mode": mode,
        "note": "No API key is stored. Samples are API-level structured transitions derived from official RoboCasa atomic task semantics, not MuJoCo rollouts.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if mode == "all":
        run_no_domain()
        run_domain(False)
        run_domain(True)
    elif mode == "candidate":
        run_domain(True)
    elif mode == "feedback":
        run_domain(True, feedback_driven=True)
    else:
        raise ValueError(f"Unsupported PROBE_MODE: {mode}")
    print(f"TRACE_DIR={OUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr, flush=True)
        raise
