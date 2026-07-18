#!/usr/bin/env python3
"""V1: grouped-action Pong contact evaluation.

A fresh Domain receives paired collision experiences designed to distinguish
the paddle action effect from per-action sampling bias. All evaluation queries
are deterministic and disable exploration and memory updates.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path


BASE_URL = os.environ.get(
    "ADAPT1_BASE_URL", "https://rei-neuroadapt-api-uat.reilabs.org"
)
PLAYER_X = 0.06
DT = 0.46
PADDLE_HALF_HEIGHT = 0.105
PADDLE_STEP = 0.105
ACTIONS = ("move_up", "move_down", "stay")
ACTION_DELTA = {"move_up": -PADDLE_STEP, "move_down": PADDLE_STEP, "stay": 0.0}


class Client:
    def __init__(self, token: str, interval: float, base_url: str = BASE_URL) -> None:
        self.token = token
        self.interval = interval
        self.base_url = base_url.rstrip("/")
        self.last_finished = 0.0

    def request(
        self,
        method: str,
        path: str,
        payload: dict,
        *,
        retries: int = 0,
        timeout: int = 90,
    ) -> dict:
        wait = self.interval - (time.monotonic() - self.last_finished)
        if wait > 0:
            time.sleep(wait)
        data = json.dumps(payload, separators=(",", ":")).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "adapt1-pong-collision-followup/1.0",
            },
        )
        try:
            for attempt in range(retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        return json.loads(response.read().decode())
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode(errors="replace")[:1200]
                    if exc.code not in (429, 502, 503, 504) or attempt == retries:
                        raise RuntimeError(
                            f"{method} {path} returned HTTP {exc.code}: {detail}"
                        ) from exc
                except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                    if attempt == retries:
                        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
                time.sleep(2.0 * (attempt + 1))
        finally:
            self.last_finished = time.monotonic()
        raise AssertionError("unreachable")


def base_state(seed: int) -> dict:
    rng = random.Random(seed)
    speed = rng.uniform(0.30, 0.40)
    crossing_time = DT * rng.uniform(0.15, 0.88)
    vy = rng.uniform(-0.32, 0.32)
    if abs(vy) < 0.04:
        vy = math.copysign(0.04, vy or 1.0)
    contact_y = rng.uniform(0.40, 0.60)
    return {
        "ball_x": PLAYER_X + speed * crossing_time,
        "ball_y": contact_y - vy * crossing_time,
        "ball_vx": -speed,
        "ball_vy": vy,
        "contact_y": contact_y,
    }


def row_for_post_offset(base: dict, action: str, post_offset: float, design: str) -> dict:
    desired_post_paddle = base["contact_y"] + post_offset
    initial_paddle = desired_post_paddle - ACTION_DELTA[action]
    if not PADDLE_HALF_HEIGHT <= initial_paddle <= 1.0 - PADDLE_HALF_HEIGHT:
        raise ValueError("generated initial paddle is out of bounds")
    return make_row(base, action, initial_paddle, design, post_offset)


def row_for_initial_offset(base: dict, action: str, initial_offset: float, design: str) -> dict:
    initial_paddle = base["contact_y"] + initial_offset
    return make_row(base, action, initial_paddle, design, initial_offset)


def make_row(base: dict, action: str, initial_paddle: float, design: str, offset: float) -> dict:
    post_paddle = min(
        1.0 - PADDLE_HALF_HEIGHT,
        max(PADDLE_HALF_HEIGHT, initial_paddle + ACTION_DELTA[action]),
    )
    post_offset = post_paddle - base["contact_y"]
    returned = abs(post_offset) <= PADDLE_HALF_HEIGHT + 1e-12
    context = {
        "ball_x": round(base["ball_x"], 6),
        "ball_y": round(base["ball_y"], 6),
        "ball_vx": round(base["ball_vx"], 6),
        "ball_vy": round(base["ball_vy"], 6),
        "paddle_center_y": round(initial_paddle, 6),
        "executed_action": action,
    }
    truth = {
        "next_ball_vx": round(-base["ball_vx"] if returned else base["ball_vx"], 6),
        "next_paddle_center_y": round(post_paddle, 6),
        "native_outcome": "paddle_return" if returned else "paddle_miss",
    }
    return {
        "design": design,
        "offset": round(offset, 6),
        "post_offset": round(post_offset, 6),
        "contact_y": round(base["contact_y"], 6),
        "action": action,
        "context": context,
        "truth": truth,
    }


def build_training() -> list[dict]:
    rows = []
    aligned_offsets = (-0.18, -0.09, 0.0, 0.09, 0.18)
    for i in range(4):
        base = base_state(0xC0111000 + i * 7919)
        for action in ACTIONS:
            for offset in aligned_offsets:
                rows.append(row_for_post_offset(base, action, offset, "aligned_post_geometry"))
    for i in range(5):
        base = base_state(0xC0112000 + i * 104729)
        for action in ACTIONS:
            for offset in (-0.08, 0.08):
                rows.append(row_for_initial_offset(base, action, offset, "same_initial_state"))
    random.Random(0xC011FACE).shuffle(rows)
    return rows


def build_evaluation() -> list[dict]:
    rows = []
    aligned_offsets = (-0.16, -0.11, -0.08, 0.0, 0.08, 0.11, 0.16)
    for i in range(3):
        base = base_state(0xE7A11000 + i * 15485863)
        for action in ACTIONS:
            for offset in aligned_offsets:
                rows.append(row_for_post_offset(base, action, offset, "aligned_post_geometry"))
    for i in range(3):
        base = base_state(0xE7A12000 + i * 32452843)
        for action in ACTIONS:
            for offset in (-0.08, 0.08):
                rows.append(row_for_initial_offset(base, action, offset, "same_initial_state"))
    return rows


def query_payload(session_id: str, relation: str, context: dict) -> dict:
    return {
        "session_id": session_id,
        "question": "Predict the immediate collision outcome and next values for this Pong state and action.",
        "context": context,
        "relation": relation,
        "selection_mode": "deterministic",
        "allow_exploration": False,
        "update_memory_state": False,
        "return_fields": ["transition_prediction", "learning_state"],
        "top_k": 10,
    }


def parse_prediction(response: dict) -> dict:
    prediction = response.get("transition_prediction") or {}
    values = prediction.get("values") or (prediction.get("output") or {}).get("values") or {}
    if prediction.get("status") != "predicted":
        return {
            "status": prediction.get("status", "absent"),
            "reason": prediction.get("abstention_reason"),
        }
    return {
        "status": "predicted",
        "values": values,
        "confidence": prediction.get("confidence"),
        "support_count": prediction.get("support_count"),
        "nearest_distance": prediction.get("nearest_distance"),
        "mean_distance": prediction.get("mean_distance"),
        "model_version": prediction.get("model_version"),
    }


def evaluate(client: Client, domain_id: str, session_id: str, relation: str, rows: list[dict], label: str, emit) -> list[dict]:
    evaluated = []
    for index, row in enumerate(rows, 1):
        response = client.request(
            "POST",
            f"/api/v1/domains/{domain_id}/query",
            query_payload(session_id, relation, row["context"]),
            retries=3,
        )
        record = {**row, "source": label, "prediction": parse_prediction(response)}
        evaluated.append(record)
        emit({"type": "evaluation", **record})
        if index % 15 == 0 or index == len(rows):
            print(f"{label}: evaluated {index}/{len(rows)}", flush=True)
    return evaluated


def summary(rows: list[dict]) -> dict:
    covered = [r for r in rows if r["prediction"].get("status") == "predicted"]
    correct = [
        r for r in covered
        if str(r["prediction"]["values"].get("native_outcome")) == r["truth"]["native_outcome"]
    ]
    vx = [
        abs(float(r["prediction"]["values"]["next_ball_vx"]) - r["truth"]["next_ball_vx"])
        for r in covered if "next_ball_vx" in r["prediction"]["values"]
    ]
    paddle = [
        abs(float(r["prediction"]["values"]["next_paddle_center_y"]) - r["truth"]["next_paddle_center_y"])
        for r in covered if "next_paddle_center_y" in r["prediction"]["values"]
    ]
    confusion = Counter(
        (r["truth"]["native_outcome"], str(r["prediction"]["values"].get("native_outcome")))
        for r in covered
    )
    by_design = {}
    for design in sorted({r["design"] for r in rows}):
        part = [r for r in covered if r["design"] == design]
        by_design[design] = {
            "rows": len(part),
            "outcome_accuracy": sum(
                str(r["prediction"]["values"].get("native_outcome")) == r["truth"]["native_outcome"]
                for r in part
            ) / len(part) if part else None,
        }
    by_abs_post_offset = {}
    for value in sorted({round(abs(r["post_offset"]), 3) for r in rows}):
        part = [r for r in covered if round(abs(r["post_offset"]), 3) == value]
        by_abs_post_offset[str(value)] = {
            "rows": len(part),
            "truth": dict(Counter(r["truth"]["native_outcome"] for r in part)),
            "predicted": dict(Counter(str(r["prediction"]["values"].get("native_outcome")) for r in part)),
            "accuracy": sum(
                str(r["prediction"]["values"].get("native_outcome")) == r["truth"]["native_outcome"]
                for r in part
            ) / len(part) if part else None,
        }
    consistency_groups = defaultdict(list)
    for r in covered:
        if r["design"] == "aligned_post_geometry":
            key = (
                r["context"]["ball_x"], r["context"]["ball_y"],
                r["context"]["ball_vx"], r["context"]["ball_vy"], r["post_offset"],
            )
            consistency_groups[key].append(str(r["prediction"]["values"].get("native_outcome")))
    consistent = sum(len(set(values)) == 1 for values in consistency_groups.values())
    confidences = [r["prediction"].get("confidence") for r in covered]
    confidences = [float(x) for x in confidences if isinstance(x, (int, float))]
    return {
        "rows": len(rows),
        "coverage": len(covered) / len(rows) if rows else 0.0,
        "outcome_accuracy": len(correct) / len(covered) if covered else None,
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
        "next_ball_vx_mae": statistics.fmean(vx) if vx else None,
        "next_paddle_center_y_mae": statistics.fmean(paddle) if paddle else None,
        "mean_confidence": statistics.fmean(confidences) if confidences else None,
        "by_design": by_design,
        "by_abs_post_offset": by_abs_post_offset,
        "aligned_cross_action_consistency": consistent / len(consistency_groups) if consistency_groups else None,
        "aligned_consistency_groups": len(consistency_groups),
    }


def make_emitter(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    def emit(record: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    return emit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--results", default="pong-collision-followup-results.jsonl")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    token = getpass.getpass("Adapt-1 API key (hidden): ").strip()
    if not token:
        parser.error("API key required")
    client = Client(token, args.interval, args.base_url)
    training = build_training()
    evaluation = build_evaluation()
    run_id = str(uuid.uuid4())
    domain_id = "pong-collision-crossed-" + run_id[:12]
    session_id = "pong-collision-session-" + run_id
    result_path = Path(args.results).resolve()
    emit = make_emitter(result_path)
    emit({
        "type": "design",
        "version": "v1_grouped_action",
        "domain_id": domain_id,
        "session_id": session_id,
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
    })
    relation = "pong_collision_dynamics"
    create_payload = {
        "domain_id": domain_id,
        "description": (
            "Learn immediate Pong paddle-collision consequences from executed interventions. "
            "Coordinates are normalized, y increases downward, move_up subtracts one paddle "
            "actuator step, move_down adds one step, and stay leaves the paddle fixed. Each event "
            "binds the current visible ball and paddle state plus the executed action to the observed "
            "horizontal velocity, resulting paddle center, and native return-or-miss outcome. The "
            "collision boundary, correct action, projected contact point, and solution are not supplied."
        ),
        "schema": {
            "entities": ["ball", "player_paddle"],
            "relations": [relation],
            "signals": [
                "ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_center_y",
                "executed_action", "next_ball_vx", "next_paddle_center_y", "native_outcome",
            ],
            "event_types": ["collision_transition"],
            "constraints": {},
        },
        "hypotheses": [],
        "query_templates": {},
        "learning": {
            "enabled": True,
            "transition": {
                "enabled": True,
                "event_types": ["collision_transition"],
                "input_paths": [
                    "values.ball_x", "values.ball_y", "values.ball_vx",
                    "values.ball_vy", "values.paddle_center_y",
                ],
                "group_by_paths": ["values.executed_action"],
                "targets": [
                    {"path": "values.next_ball_vx", "type": "number"},
                    {"path": "values.next_paddle_center_y", "type": "number"},
                    {"path": "values.native_outcome", "type": "categorical"},
                ],
                "required_support": 2,
                "max_samples": 512,
                "neighbors": 8,
                "max_distance": 1.0,
            },
        },
    }
    created = client.request("POST", "/api/v1/domains", create_payload, timeout=120)
    emit({"type": "domain_created", "domain_id": domain_id, "response": created})
    print(f"created {domain_id}", flush=True)

    pre = client.request(
        "POST", f"/api/v1/domains/{domain_id}/query",
        query_payload(session_id, relation, evaluation[0]["context"]), retries=3,
    )
    emit({"type": "pretraining_probe", "prediction": parse_prediction(pre)})
    print("pretraining=" + json.dumps(parse_prediction(pre), sort_keys=True), flush=True)
    if parse_prediction(pre).get("status") == "predicted":
        raise RuntimeError("cold Domain unexpectedly predicted before evidence")

    for index, row in enumerate(training, 1):
        event_payload = {
            "session_id": session_id,
            "event_type": "collision_transition",
            "values": {**row["context"], **row["truth"]},
            "metadata": {
                "benchmark_id": "pong-collision-crossed-v1",
                "transition_index": index - 1,
                "design": row["design"],
            },
        }
        response = client.request(
            "POST", f"/api/v1/domains/{domain_id}/events", event_payload, timeout=90
        )
        eligibility = response.get("learner_eligibility") or {}
        if response.get("accepted") is not True or eligibility.get("accepted") is not True:
            raise RuntimeError(f"event {index} rejected: {response}")
        if index % 15 == 0 or index == len(training):
            print(
                f"fresh_domain: trained {index}/{len(training)} "
                f"version={eligibility.get('model_version')}", flush=True
            )
            emit({
                "type": "training_progress", "accepted": index,
                "sample_count": eligibility.get("sample_count"),
                "model_version": eligibility.get("model_version"),
            })

    fresh_rows = evaluate(
        client, domain_id, session_id, relation, evaluation, "fresh_crossed_domain", emit
    )
    fresh_summary = summary(fresh_rows)
    emit({"type": "summary", "source": "fresh_crossed_domain", "summary": fresh_summary})
    final = {
        "type": "final",
        "domain_id": domain_id,
        "session_id": session_id,
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "fresh_summary": fresh_summary,
    }
    emit(final)
    print("FINAL=" + json.dumps(final, sort_keys=True), flush=True)
    print(f"results={result_path}", flush=True)


if __name__ == "__main__":
    main()
