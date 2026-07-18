#!/usr/bin/env python3
"""Fresh Pong contact Domain with shared numeric action representation.

This replays the exact 162 experiences used by the grouped-action Domain but
changes the executable Domain contract:

* collision transitions have their own event type;
* action_delta_y is a numeric learner input instead of a group boundary;
* delta_t is explicit;
* three neighbors are used to reduce boundary smoothing.

No contact threshold, projected contact coordinate, preferred action, or
solution is supplied. Evaluation is write-free and uses the identical 81 rows.
"""

from __future__ import annotations

import argparse
import getpass
import json
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import pong_collision_followup as P
import pong_collision_phase2 as P2


RELATION = "pong_contact_dynamics"
EVENT_TYPE = "player_contact_transition"
BENCHMARK_ID = "pong-domain-v2-shared-action-contact-v1"


def augment_context(row: dict) -> dict:
    return {
        **row["context"],
        "action_delta_y": P.ACTION_DELTA[row["action"]],
        "delta_t": P.DT,
    }


def query_payload(session_id: str, context: dict) -> dict:
    return {
        "session_id": session_id,
        "question": (
            "Predict the immediate player-contact consequence and resulting "
            "Pong values for this state and actuator command."
        ),
        "context": context,
        "relation": RELATION,
        "selection_mode": "deterministic",
        "allow_exploration": False,
        "update_memory_state": False,
        "return_fields": ["transition_prediction", "learning_state"],
        "top_k": 10,
    }


def make_emitter(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    def emit(record: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    return emit


def support_summary(evaluated: list[dict]) -> dict:
    predictions = [r["full_prediction"] for r in evaluated]
    evidence_counts = [len(p.get("evidence_memory_ids") or []) for p in predictions]
    versions = Counter(p.get("model_version") for p in predictions)
    distances = [
        float(p["nearest_distance"])
        for p in predictions if isinstance(p.get("nearest_distance"), (int, float))
    ]
    return {
        "evidence_count_distribution": dict(Counter(evidence_counts)),
        "model_versions": {str(k): v for k, v in versions.items()},
        "mean_nearest_distance": statistics.fmean(distances) if distances else None,
    }


def action_prediction_summary(evaluated: list[dict]) -> dict:
    by_action = {}
    for action in P.ACTIONS:
        rows = [r for r in evaluated if r["action"] == action]
        by_action[action] = {
            "rows": len(rows),
            "accuracy": sum(
                r["prediction"]["values"].get("native_outcome") == r["truth"]["native_outcome"]
                for r in rows
            ) / len(rows),
            "truth": dict(Counter(r["truth"]["native_outcome"] for r in rows)),
            "predicted": dict(
                Counter(r["prediction"]["values"].get("native_outcome") for r in rows)
            ),
        }
    return by_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--base-url", default=P.BASE_URL)
    parser.add_argument("--results", default="pong-domain-v2-shared-action-results.jsonl")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    token = getpass.getpass("Adapt-1 API key (hidden): ").strip()
    if not token:
        parser.error("API key required")

    client = P.Client(token, args.interval, args.base_url)
    phase1 = P.build_training()
    phase2 = P2.build_phase2_training()
    training = phase1 + phase2
    evaluation = P.build_evaluation()
    run_id = str(uuid.uuid4())
    domain_id = "pong-domain-v2-" + run_id[:12]
    session_id = "pong-domain-v2-session-" + run_id
    result_path = Path(args.results).resolve()
    emit = make_emitter(result_path)
    emit({
        "type": "design",
        "version": "v2_numeric_action",
        "domain_id": domain_id,
        "session_id": session_id,
        "training_rows": len(training),
        "phase1_rows": len(phase1),
        "phase2_rows": len(phase2),
        "evaluation_rows": len(evaluation),
        "differences": {
            "event_type": EVENT_TYPE,
            "numeric_action_input": "action_delta_y",
            "group_by_paths": [],
            "explicit_delta_t": True,
            "neighbors": 3,
        },
    })

    create_payload = {
        "domain_id": domain_id,
        "description": (
            "Learn immediate Pong player-contact dynamics from executed actuator interventions. "
            "All events are player-contact candidates rather than free-flight transitions. "
            "Coordinates are normalized, y increases downward, delta_t is the control interval, "
            "and action_delta_y is the signed paddle actuator displacement: negative moves up, "
            "positive moves down, and zero stays. Each event binds the current visible ball and "
            "paddle state plus that actuator signal to the observed next horizontal ball velocity, "
            "resulting paddle center, and native return-or-miss outcome. The projected contact "
            "coordinate, collision threshold, correct action, and solution are not supplied."
        ),
        "schema": {
            "entities": ["ball", "player_paddle", "control_interval"],
            "relations": [RELATION],
            "signals": [
                "ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_center_y",
                "executed_action", "action_delta_y", "delta_t", "next_ball_vx",
                "next_paddle_center_y", "native_outcome",
            ],
            "event_types": [EVENT_TYPE],
            "constraints": {},
        },
        "hypotheses": [],
        "query_templates": {},
        "learning": {
            "enabled": True,
            "transition": {
                "enabled": True,
                "event_types": [EVENT_TYPE],
                "input_paths": [
                    "values.ball_x", "values.ball_y", "values.ball_vx",
                    "values.ball_vy", "values.paddle_center_y",
                    "values.action_delta_y", "values.delta_t",
                ],
                "group_by_paths": [],
                "targets": [
                    {"path": "values.next_ball_vx", "type": "number"},
                    {"path": "values.next_paddle_center_y", "type": "number"},
                    {"path": "values.native_outcome", "type": "categorical"},
                ],
                "required_support": 2,
                "max_samples": 512,
                "neighbors": 3,
                "max_distance": 1.0,
            },
        },
    }
    created = client.request("POST", "/api/v1/domains", create_payload, timeout=120)
    emit({"type": "domain_created", "response": created})
    print(f"created {domain_id}", flush=True)

    probe_context = augment_context(evaluation[0])
    pre_response = client.request(
        "POST", f"/api/v1/domains/{domain_id}/query",
        query_payload(session_id, probe_context), retries=3,
    )
    pre = P.parse_prediction(pre_response)
    emit({
        "type": "pretraining_probe",
        "prediction": pre,
        "full_transition_prediction": pre_response.get("transition_prediction"),
        "learning_state": pre_response.get("learning_state"),
    })
    print("pretraining=" + json.dumps(pre, sort_keys=True), flush=True)
    if pre.get("status") == "predicted":
        raise RuntimeError("cold Domain unexpectedly predicted before evidence")

    for index, row in enumerate(training, 1):
        context = augment_context(row)
        payload = {
            "session_id": session_id,
            "event_type": EVENT_TYPE,
            "values": {**context, **row["truth"]},
            "metadata": {
                "benchmark_id": BENCHMARK_ID,
                "transition_index": index - 1,
                "phase": 1 if index <= len(phase1) else 2,
                "design": row["design"],
            },
        }
        response = client.request(
            "POST", f"/api/v1/domains/{domain_id}/events", payload, timeout=90
        )
        eligibility = response.get("learner_eligibility") or {}
        if response.get("accepted") is not True or eligibility.get("accepted") is not True:
            raise RuntimeError(f"event {index} rejected: {response}")
        emit({
            "type": "training_event",
            "index": index,
            "action": row["action"],
            "design": row["design"],
            "post_offset": row["post_offset"],
            "truth": row["truth"],
            "response": response,
        })
        if index % 18 == 0 or index == len(training):
            print(
                f"trained {index}/{len(training)} version={eligibility.get('model_version')}",
                flush=True,
            )

    evaluated = []
    for index, row in enumerate(evaluation, 1):
        context = augment_context(row)
        response = client.request(
            "POST", f"/api/v1/domains/{domain_id}/query",
            query_payload(session_id, context), retries=3,
        )
        prediction = P.parse_prediction(response)
        if prediction.get("status") != "predicted":
            raise RuntimeError(f"evaluation query {index} abstained: {prediction}")
        record = {
            **row,
            "source": "domain_v2_shared_action",
            "prediction": prediction,
            "full_prediction": response.get("transition_prediction") or {},
            "learning_state": response.get("learning_state"),
        }
        evaluated.append(record)
        emit({"type": "evaluation", **record})
        if index % 15 == 0 or index == len(evaluation):
            print(f"evaluated {index}/{len(evaluation)}", flush=True)

    result = {
        "type": "final",
        "domain_id": domain_id,
        "session_id": session_id,
        "model_version": len(training),
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "summary": P.summary(evaluated),
        "by_action": action_prediction_summary(evaluated),
        "support_diagnostics": support_summary(evaluated),
    }
    emit(result)
    print("FINAL=" + json.dumps(result, sort_keys=True), flush=True)
    print(f"results={result_path}", flush=True)


if __name__ == "__main__":
    main()
