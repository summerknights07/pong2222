#!/usr/bin/env python3
"""Pong collision Domain using the observable post-action paddle state.

This Domain isolates contact dynamics from actuator dynamics.  The paddle command
has already executed, so the learner receives the resulting paddle center as an
observable state variable.  Collision evidence can therefore pool across the
original action labels.  No projected ball-contact coordinate or collision rule
is supplied.

The script is safely resumable: rerunning it against its existing results file
checks the live model version before continuing from the next missing event.
"""

from __future__ import annotations

import argparse
import getpass
import json
import statistics
import uuid
from collections import Counter
from pathlib import Path

import pong_collision_followup as P
import pong_collision_phase2 as P2
import pong_domain_v2_shared_action as V2


RELATION = "pong_player_contact"
EVENT_TYPE = "post_actuation_contact_transition"
BENCHMARK_ID = "pong-domain-v3-post-action-contact-v1"


def contact_context(row: dict) -> dict:
    return {
        "ball_x": row["context"]["ball_x"],
        "ball_y": row["context"]["ball_y"],
        "ball_vx": row["context"]["ball_vx"],
        "ball_vy": row["context"]["ball_vy"],
        "post_action_paddle_center_y": row["truth"]["next_paddle_center_y"],
        "delta_t": P.DT,
        # Retained for auditability but deliberately absent from input_paths.
        "executed_action": row["action"],
    }


def query_payload(session_id: str, context: dict) -> dict:
    return {
        "session_id": session_id,
        "question": (
            "Predict the immediate player-contact consequence from the incoming "
            "ball state and observed post-actuation paddle state."
        ),
        "context": context,
        "relation": RELATION,
        "selection_mode": "deterministic",
        "allow_exploration": False,
        "update_memory_state": False,
        "return_fields": ["transition_prediction", "learning_state"],
        "top_k": 10,
    }


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def live_version(response: dict) -> int | None:
    state = response.get("learning_state") or {}
    subsystem = ((state.get("subsystems") or {}).get("structured_transition") or {})
    return subsystem.get("model_version")


def evidence_audit(evaluated: list[dict], memory_map: dict[str, dict]) -> dict:
    same_action_fractions = []
    action_set_sizes = []
    missing = 0
    for row in evaluated:
        evidence = []
        for memory_id in row["full_prediction"].get("evidence_memory_ids") or []:
            event = memory_map.get(memory_id)
            if event is None:
                missing += 1
            else:
                evidence.append(event)
        if evidence:
            same_action_fractions.append(
                sum(event["action"] == row["action"] for event in evidence) / len(evidence)
            )
            action_set_sizes.append(len({event["action"] for event in evidence}))
    return {
        "mapped_evidence_ids": sum(
            len(row["full_prediction"].get("evidence_memory_ids") or []) for row in evaluated
        ) - missing,
        "missing_evidence_ids": missing,
        "mean_same_action_support_fraction": (
            statistics.fmean(same_action_fractions) if same_action_fractions else None
        ),
        "support_action_set_size_distribution": dict(Counter(action_set_sizes)),
        "queries_with_cross_action_support": sum(size > 1 for size in action_set_sizes),
    }


def create_domain(client: P.Client, domain_id: str) -> dict:
    payload = {
        "domain_id": domain_id,
        "description": (
            "Learn immediate Pong player-contact consequences after the paddle actuator has "
            "executed. Every event is a player-contact candidate. Coordinates are normalized, "
            "y increases downward, delta_t is the control interval, and "
            "post_action_paddle_center_y is the observed resulting paddle state. Events bind "
            "the incoming visible ball state and resulting paddle state to the observed next "
            "horizontal ball velocity and native return-or-miss outcome. Original action names "
            "are retained only as audit metadata and do not partition learning. The projected "
            "ball-contact coordinate, paddle collision threshold, correct action, and solution "
            "are not supplied."
        ),
        "schema": {
            "entities": ["ball", "player_paddle", "control_interval"],
            "relations": [RELATION],
            "signals": [
                "ball_x", "ball_y", "ball_vx", "ball_vy",
                "post_action_paddle_center_y", "delta_t", "executed_action",
                "next_ball_vx", "native_outcome",
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
                    "values.ball_vy", "values.post_action_paddle_center_y",
                    "values.delta_t",
                ],
                "group_by_paths": [],
                "targets": [
                    {"path": "values.next_ball_vx", "type": "number"},
                    {"path": "values.native_outcome", "type": "categorical"},
                ],
                "required_support": 2,
                "max_samples": 512,
                "neighbors": 8,
                "max_distance": 1.0,
            },
        },
    }
    return client.request("POST", "/api/v1/domains", payload, timeout=120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--base-url", default=P.BASE_URL)
    parser.add_argument("--results", default="pong-domain-v3-contact-state-results.jsonl")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    token = getpass.getpass("Adapt-1 API key (hidden): ").strip()
    if not token:
        parser.error("API key required")

    path = Path(args.results).resolve()
    records = read_records(path)
    phase1 = P.build_training()
    phase2 = P2.build_phase2_training()
    training = phase1 + phase2
    evaluation = P.build_evaluation()
    client = P.Client(token, args.interval, args.base_url)

    if not records:
        run_id = str(uuid.uuid4())
        domain_id = "pong-domain-v3-" + run_id[:12]
        session_id = "pong-domain-v3-session-" + run_id
        path.write_text("", encoding="utf-8")
        append(path, {
            "type": "design",
            "version": "v3_post_action_state",
            "domain_id": domain_id,
            "session_id": session_id,
            "training_rows": len(training),
            "evaluation_rows": len(evaluation),
            "differences": {
                "input_state": "post_action_paddle_center_y",
                "executed_action_is_learner_input": False,
                "group_by_paths": [],
                "contact_only": True,
                "neighbors": 8,
            },
        })
        created = create_domain(client, domain_id)
        append(path, {"type": "domain_created", "response": created})
        print(f"created {domain_id}", flush=True)
        probe = client.request(
            "POST", f"/api/v1/domains/{domain_id}/query",
            query_payload(session_id, contact_context(evaluation[0])), retries=3,
        )
        parsed = P.parse_prediction(probe)
        append(path, {
            "type": "pretraining_probe", "prediction": parsed,
            "full_transition_prediction": probe.get("transition_prediction"),
            "learning_state": probe.get("learning_state"),
        })
        print("pretraining=" + json.dumps(parsed, sort_keys=True), flush=True)
        if parsed.get("status") == "predicted":
            raise RuntimeError("cold Domain unexpectedly predicted before evidence")
        records = read_records(path)

    design = next(record for record in records if record.get("type") == "design")
    domain_id = design["domain_id"]
    session_id = design["session_id"]
    logged_events = [r for r in records if r.get("type") == "training_event"]
    if any(r.get("type") == "final" for r in records):
        raise RuntimeError("run is already complete")

    check = client.request(
        "POST", f"/api/v1/domains/{domain_id}/query",
        query_payload(session_id, contact_context(evaluation[0])), retries=3,
    )
    version = live_version(check)
    print(f"state_check logged={len(logged_events)} live_version={version}", flush=True)
    if version != len(logged_events):
        raise RuntimeError(
            f"refusing unsafe resume: logged {len(logged_events)} but live version is {version}"
        )
    append(path, {
        "type": "state_check", "logged_events": len(logged_events),
        "live_model_version": version,
    })

    for zero_index in range(len(logged_events), len(training)):
        row = training[zero_index]
        index = zero_index + 1
        context = contact_context(row)
        payload = {
            "session_id": session_id,
            "event_type": EVENT_TYPE,
            "values": {**context, "next_ball_vx": row["truth"]["next_ball_vx"],
                       "native_outcome": row["truth"]["native_outcome"]},
            "metadata": {
                "benchmark_id": BENCHMARK_ID,
                "transition_index": zero_index,
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
        append(path, {
            "type": "training_event", "index": index, "action": row["action"],
            "design": row["design"], "post_offset": row["post_offset"],
            "truth": row["truth"], "response": response,
        })
        if index % 18 == 0 or index == len(training):
            print(f"trained {index}/{len(training)} version={eligibility.get('model_version')}", flush=True)

    memory_map = {}
    for record in read_records(path):
        if record.get("type") == "training_event":
            memory_id = (record.get("response") or {}).get("memory_id")
            if memory_id:
                memory_map[memory_id] = record

    evaluated = []
    for index, row in enumerate(evaluation, 1):
        response = client.request(
            "POST", f"/api/v1/domains/{domain_id}/query",
            query_payload(session_id, contact_context(row)), retries=3,
        )
        prediction = P.parse_prediction(response)
        if prediction.get("status") != "predicted":
            raise RuntimeError(f"evaluation query {index} abstained: {prediction}")
        record = {
            **row, "source": "domain_v3_contact_state", "prediction": prediction,
            "full_prediction": response.get("transition_prediction") or {},
            "learning_state": response.get("learning_state"),
        }
        evaluated.append(record)
        append(path, {"type": "evaluation", **record})
        if index % 15 == 0 or index == len(evaluation):
            print(f"evaluated {index}/{len(evaluation)}", flush=True)

    result = {
        "type": "final", "domain_id": domain_id, "session_id": session_id,
        "model_version": len(training),
        "training_rows": len(training), "evaluation_rows": len(evaluation),
        "summary": P.summary(evaluated),
        "by_action": V2.action_prediction_summary(evaluated),
        "support_diagnostics": V2.support_summary(evaluated),
        "evidence_audit": evidence_audit(evaluated, memory_map),
    }
    append(path, result)
    print("FINAL=" + json.dumps(result, sort_keys=True), flush=True)
    print(f"results={path}", flush=True)


if __name__ == "__main__":
    main()
