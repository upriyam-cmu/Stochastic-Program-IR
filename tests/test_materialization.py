import unittest
from typing import Any, cast

import numpy as np

from stochastic_programming_library import (
    Normal,
    RandomVariable,
    SamplingCheckpoint,
    sampling_phase,
)
from stochastic_programming_library.expr.hashing import (
    resolve_stochastic_hashes,
)


class MaterializationTests(unittest.TestCase):
    def test_plate_sampling_is_repeatable_for_seed_and_key(self) -> None:
        expr = Normal(0, 1, rng_key="weights").add_plates(
            "row",
            expect=(),
        )

        first = expr.realize(seed=7, plate_sizes={"row": 4})
        second = expr.realize(seed=7, plate_sizes={"row": 4})

        np.testing.assert_array_equal(first.data, second.data)
        self.assertEqual(first.layout.plates, ("row",))

    def test_partial_materialization_reuses_earlier_phase(self) -> None:
        with sampling_phase("latent"):
            latent = Normal(0, 1)
        with sampling_phase("observation"):
            observation = Normal(latent, 1).add_plates(
                "row",
                expect=(),
            )

        checkpoint = observation.materialize(
            seed=1,
            plate_sizes={"row": 4},
            phases=("latent",),
        )
        first = checkpoint.materialize(
            seed=2,
            phases=("observation",),
        )
        second = checkpoint.materialize(
            seed=3,
            phases=("observation",),
        )

        self.assertEqual(checkpoint.pending_phases, frozenset({"observation"}))
        self.assertTrue(first.is_fully_materialized)
        self.assertTrue(second.is_fully_materialized)
        self.assertFalse(np.array_equal(first.realize().data, second.realize().data))

    def test_cleared_phase_completes_when_dependency_later_resolves(self) -> None:
        with sampling_phase("latent"):
            latent = Normal(0, 1)
        with sampling_phase("observation"):
            observation = Normal(latent, 1)

        waiting = observation.materialize(
            seed=1,
            phases=("observation",),
        )
        completed = waiting.materialize(
            seed=2,
            phases=("latent",),
        )

        self.assertEqual(waiting.pending_phases, frozenset({"latent"}))
        self.assertTrue(completed.is_fully_materialized)

    def test_aliasing_changes_stochastic_but_not_structural_equality(self) -> None:
        with sampling_phase("draw"):
            shared = Normal(0, 1)
            aliased = shared + shared
        with sampling_phase("draw"):
            independent = Normal(0, 1) + Normal(0, 1)

        aliased_checkpoint = aliased.materialize(seed=5, phases=())
        independent_checkpoint = independent.materialize(seed=5, phases=())

        self.assertTrue(aliased.structurally_equal(independent))
        self.assertFalse(
            aliased_checkpoint.stochastically_equal(independent_checkpoint)
        )

    def test_equivalent_allocations_resolve_equivalent_keys(self) -> None:
        with sampling_phase("draw"):
            left = Normal(0, 1) + Normal(0, 1)
        with sampling_phase("draw"):
            right_operand = Normal(0, 1)
            left_operand = Normal(0, 1)
            right = left_operand + right_operand

        left_checkpoint = left.materialize(seed=5, phases=())
        right_checkpoint = right.materialize(seed=5, phases=())

        self.assertTrue(left_checkpoint.stochastically_equal(right_checkpoint))

    def test_symmetric_nodes_receive_distinct_graph_keys(self) -> None:
        with sampling_phase("draw"):
            left = Normal(0, 1)
            right = Normal(0, 1)
            left_consumer = Normal(left, 1)
            right_consumer = Normal(right, 1)
            independent = left_consumer + right_consumer

        resolved = resolve_stochastic_hashes(independent)

        self.assertNotEqual(
            resolved.rng_key_for(left),
            resolved.rng_key_for(right),
        )
        self.assertEqual(
            {
                part.enumeration.ordinal
                for part in (
                    resolved.for_node(left),
                    resolved.for_node(right),
                )
            },
            {0, 1},
        )

    def test_aliases_share_one_projected_stochastic_node(self) -> None:
        with sampling_phase("draw"):
            shared = Normal(0, 1)
            aliased = shared + shared

        resolved = resolve_stochastic_hashes(aliased)

        self.assertEqual(resolved.projection.nodes, (shared,))

    def test_checkpoint_is_opaque_and_not_composable(self) -> None:
        with sampling_phase("draw"):
            expr = Normal(0, 1)

        checkpoint = expr.materialize(seed=1, phases=())

        self.assertIsInstance(checkpoint, SamplingCheckpoint)
        self.assertNotIsInstance(checkpoint, RandomVariable)
        self.assertFalse(hasattr(expr, "stochastically_equal"))
        with self.assertRaises(TypeError):
            _ = cast(Any, checkpoint) + 1


if __name__ == "__main__":
    unittest.main()
