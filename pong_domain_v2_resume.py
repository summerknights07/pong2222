#!/usr/bin/env python3
"""Safely resume the v2 Domain after an ambiguous staging transport failure."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path

import pong_collision_followup as P
import pong_collision_phase2 as P2
import pong_domain_v2_shared_action as V2


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def existing_event_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "training_event":
                count += 1
    return count


def run_identifiers(path: Path) -> tuple[str | None, str | None]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "design":
                return record.get("domain_id"), record.get("session_id")
    return None, None


def live_version(response: dict) -> int | None:
    state = response.get("learning_state") or {}
    subsystem = ((state.get("subsystems") or {}).get("structured_transition") or {})
    return subsystem.get("model_version")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id")
    parser.add_argument("--session-id")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--base-url", default=P.BASE_URL)
    parser.add_argument("--results", default="pong-domain-v2-shared-action-results.jsonl")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    token = getpass.getpass("Adapt-1 API key (hidden): ").strip()
    if not token:
        parser.error("API key required")

    path = Path(args.results).resolve()
    if not path.exists():
        parser.error(f"results file does not exist: {path}")
    saved_domain_id, saved_session_id = run_identifiers(path)
    domain_id = args.domain_id or saved_domain_id
    session_id = args.session_id or saved_session_id
    if not domain_id or not session_id:
        parser.error("domain/session IDs are required or must exist in the design record")
    logged = existing_event_count(path)
    training = P.build_training() + P2.build_phase2_training()
    evaluation = P.build_evaluation()
    if logged >= len(training):
        raise RuntimeError(f"nothing to resume: {logged}/{len(training)} events already logged")

    client = P.Client(token, args.interval, args.base_url)
    check = client.request(
        "POST", f"/api/v1/domains/{domain_id}/query",
        V2.query_payload(session_id, V2.augment_context(evaluation[0])), retries=3,
    )
    version = live_version(check)
    print(f"recovery_check logged={logged} live_version={version}", flush=True)
    if version != logged:
        raise RuntimeError(
            f"refusing unsafe resume: logged {logged} events but live model is version {version}"
        )
    append_record(path, {
        "type": "recovery_check",
        "logged_events": logged,
        "live_model_version": version,
        "full_transition_prediction": check.get("transition_prediction"),
    })

    for zero_index in range(logged, len(training)):
        row = training[zero_index]
        index = zero_index + 1
        context = V2.augment_context(row)
        payload = {
            "session_id": session_id,
            "event_type": V2.EVENT_TYPE,
            "values": {**context, **row["truth"]},
            "metadata": {
                "benchmark_id": V2.BENCHMARK_ID,
                "transition_index": zero_index,
                "phase": 1 if index <= len(P.build_training()) else 2,
                "design": row["design"],
            },
        }
        response = client.request(
            "POST", f"/api/v1/domains/{domain_id}/events", payload, timeout=90
        )
        eligibility = response.get("learner_eligibility") or {}
        if response.get("accepted") is not True or eligibility.get("accepted") is not True:
            raise RuntimeError(f"event {index} rejected: {response}")
        append_record(path, {
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
        context = V2.augment_context(row)
        response = client.request(
            "POST", f"/api/v1/domains/{domain_id}/query",
            V2.query_payload(session_id, context), retries=3,
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
        append_record(path, {"type": "evaluation", **record})
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
        "by_action": V2.action_prediction_summary(evaluated),
        "support_diagnostics": V2.support_summary(evaluated),
    }
    append_record(path, result)
    print("FINAL=" + json.dumps(result, sort_keys=True), flush=True)
    print(f"results={path}", flush=True)


if __name__ == "__main__":
    main()
