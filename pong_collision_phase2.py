#!/usr/bin/env python3
"""Continue the crossed Pong Domain with boundary-bracketing experiences."""

from __future__ import annotations

import argparse
import getpass
import json
import random
from pathlib import Path

import pong_collision_followup as P


RELATION = "pong_collision_dynamics"
EVENT_TYPE = "collision_transition"
DEFAULT_SESSION_ID = "pong-collision-phase2-session"


def build_phase2_training() -> list[dict]:
    rows = []
    # The true rule is not supplied.  These are observed interventions on new
    # trajectories that bracket the unresolved interval from phase 1.
    offsets = (-0.15, -0.11, -0.10, 0.10, 0.11, 0.15)
    for i in range(4):
        base = P.base_state(0xC0113000 + i * 49979687)
        for action in P.ACTIONS:
            for offset in offsets:
                rows.append(P.row_for_post_offset(base, action, offset, "boundary_bracketing"))
    random.Random(0xB0A7DA12).shuffle(rows)
    return rows


def make_emitter(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    def emit(record: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    return emit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--base-url", default=P.BASE_URL)
    parser.add_argument("--results", default="pong-collision-phase2-results.jsonl")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    token = getpass.getpass("Adapt-1 API key (hidden): ").strip()
    if not token:
        parser.error("API key required")
    client = P.Client(token, args.interval, args.base_url)
    training = build_phase2_training()
    evaluation = P.build_evaluation()
    emit = make_emitter(Path(args.results).resolve())
    emit({
        "type": "design",
        "version": "v1_boundary_bracketing_phase2",
        "domain_id": args.domain_id,
        "session_id": args.session_id,
        "new_training_rows": len(training),
        "reevaluation_rows": len(evaluation),
    })

    probe_response = client.request(
        "POST",
        f"/api/v1/domains/{args.domain_id}/query",
        P.query_payload(args.session_id, RELATION, evaluation[0]["context"]),
        retries=3,
    )
    probe = P.parse_prediction(probe_response)
    emit({"type": "pre_phase2_probe", "prediction": probe})
    print("pre_phase2=" + json.dumps(probe, sort_keys=True), flush=True)
    if probe.get("model_version") != 90:
        raise RuntimeError(f"expected model version 90 before phase 2, got {probe}")

    for index, row in enumerate(training, 1):
        payload = {
            "session_id": args.session_id,
            "event_type": EVENT_TYPE,
            "values": {**row["context"], **row["truth"]},
            "metadata": {
                "benchmark_id": "pong-collision-crossed-v1-exploratory-phase2",
                "transition_index": 89 + index,
                "design": row["design"],
            },
        }
        response = client.request(
            "POST", f"/api/v1/domains/{args.domain_id}/events", payload, timeout=90
        )
        eligibility = response.get("learner_eligibility") or {}
        if response.get("accepted") is not True or eligibility.get("accepted") is not True:
            raise RuntimeError(f"phase-2 event {index} rejected: {response}")
        if index % 12 == 0 or index == len(training):
            print(
                f"phase2: trained {index}/{len(training)} "
                f"version={eligibility.get('model_version')}", flush=True
            )
            emit({
                "type": "training_progress", "accepted": index,
                "sample_count": eligibility.get("sample_count"),
                "model_version": eligibility.get("model_version"),
            })

    rows = P.evaluate(
        client, args.domain_id, args.session_id, RELATION, evaluation,
        "phase2_boundary_domain", emit
    )
    result = {
        "type": "final",
        "domain_id": args.domain_id,
        "session_id": args.session_id,
        "model_version": 90 + len(training),
        "new_training_rows": len(training),
        "summary": P.summary(rows),
    }
    emit(result)
    print("FINAL=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
