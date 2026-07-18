from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pong_collision_followup as v1
import pong_collision_phase2 as v1_phase2
import pong_domain_v2_shared_action as v2
import pong_domain_v3_contact_state as v3


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = v1.build_training()
        cls.phase2 = v1_phase2.build_phase2_training()
        cls.evaluation = v1.build_evaluation()

    def test_fixed_row_counts(self) -> None:
        self.assertEqual(len(self.phase1), 90)
        self.assertEqual(len(self.phase2), 72)
        self.assertEqual(len(self.phase1 + self.phase2), 162)
        self.assertEqual(len(self.evaluation), 81)

    def test_action_balance(self) -> None:
        self.assertEqual(Counter(row["action"] for row in self.phase1), {
            "move_up": 30, "move_down": 30, "stay": 30,
        })
        self.assertEqual(Counter(row["action"] for row in self.phase2), {
            "move_up": 24, "move_down": 24, "stay": 24,
        })
        self.assertEqual(Counter(row["action"] for row in self.evaluation), {
            "move_up": 27, "move_down": 27, "stay": 27,
        })

    def test_design_composition(self) -> None:
        self.assertEqual(Counter(row["design"] for row in self.phase1), {
            "aligned_post_geometry": 60,
            "same_initial_state": 30,
        })
        self.assertEqual(Counter(row["design"] for row in self.phase2), {
            "boundary_bracketing": 72,
        })
        self.assertEqual(Counter(row["design"] for row in self.evaluation), {
            "aligned_post_geometry": 63,
            "same_initial_state": 18,
        })

    def test_training_and_evaluation_states_are_disjoint(self) -> None:
        def identity(row: dict) -> tuple:
            context = row["context"]
            return (
                context["ball_x"], context["ball_y"], context["ball_vx"],
                context["ball_vy"], context["paddle_center_y"], row["action"],
            )

        training = {identity(row) for row in self.phase1 + self.phase2}
        held_out = {identity(row) for row in self.evaluation}
        self.assertTrue(training.isdisjoint(held_out))

    def test_truth_matches_hidden_boundary(self) -> None:
        for row in self.phase1 + self.phase2 + self.evaluation:
            expected = (
                "paddle_return"
                if abs(row["post_offset"]) <= v1.PADDLE_HALF_HEIGHT + 1e-12
                else "paddle_miss"
            )
            self.assertEqual(row["truth"]["native_outcome"], expected)


class RepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = v1.build_evaluation()[0]

    def test_v2_numeric_action_context(self) -> None:
        context = v2.augment_context(self.row)
        self.assertEqual(context["action_delta_y"], v1.ACTION_DELTA[self.row["action"]])
        self.assertEqual(context["delta_t"], v1.DT)

    def test_v3_uses_observed_post_action_state(self) -> None:
        context = v3.contact_context(self.row)
        self.assertEqual(
            context["post_action_paddle_center_y"],
            self.row["truth"]["next_paddle_center_y"],
        )
        self.assertEqual(context["executed_action"], self.row["action"])
        self.assertNotIn("paddle_center_y", context)

    def test_evaluation_queries_are_write_free(self) -> None:
        payloads = [
            v1.query_payload("session", "relation", self.row["context"]),
            v2.query_payload("session", v2.augment_context(self.row)),
            v3.query_payload("session", v3.contact_context(self.row)),
        ]
        for payload in payloads:
            self.assertEqual(payload["selection_mode"], "deterministic")
            self.assertFalse(payload["allow_exploration"])
            self.assertFalse(payload["update_memory_state"])


if __name__ == "__main__":
    unittest.main()
