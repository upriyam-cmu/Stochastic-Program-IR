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
from stochastic_programming_library.errors import UnrealizedGraphError


class MaterializationTests(unittest.TestCase):
    def test_plate_sampling_is_repeatable_for_seed_and_label(self) -> None:
        expr = Normal(0, 1, rng_label="weights").add_plates(
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
        self.assertFalse(np.array_equal(first.value().data, second.value().data))

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
        latent_first = observation.materialize(
            seed=2,
            phases=("latent",),
        )
        expected = latent_first.materialize(
            seed=1,
            phases=("observation",),
        )

        self.assertEqual(waiting.pending_phases, frozenset({"latent"}))
        self.assertTrue(completed.is_fully_materialized)
        np.testing.assert_array_equal(completed.value().data, expected.value().data)

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

    def test_equivalent_allocations_resolve_equivalent_entropy(self) -> None:
        with sampling_phase("draw"):
            left = Normal(0, 1) + Normal(0, 1)
        with sampling_phase("draw"):
            right_operand = Normal(0, 1)
            left_operand = Normal(0, 1)
            right = left_operand + right_operand

        left_checkpoint = left.materialize(seed=5, phases=())
        right_checkpoint = right.materialize(seed=5, phases=())

        self.assertTrue(left_checkpoint.stochastically_equal(right_checkpoint))

    def test_symmetric_nodes_receive_distinct_node_entropy(self) -> None:
        with sampling_phase("draw"):
            left = Normal(0, 1)
            right = Normal(0, 1)
            left_consumer = Normal(left, 1)
            right_consumer = Normal(right, 1)
            independent = left_consumer + right_consumer

        resolved = resolve_stochastic_hashes(independent)

        self.assertNotEqual(
            resolved.node_entropy_for(left),
            resolved.node_entropy_for(right),
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

    def test_repeated_stochastic_consumption_changes_graph_hashes(self) -> None:
        single_source = Normal(0, 1)
        single_root = Normal(single_source, 1)
        repeated_source = Normal(0, 1)
        repeated_root = Normal(repeated_source + repeated_source, 1)

        single = resolve_stochastic_hashes(single_root)
        repeated = resolve_stochastic_hashes(repeated_root)

        self.assertNotEqual(
            single.for_node(single_source).final,
            repeated.for_node(repeated_source).final,
        )
        self.assertEqual(
            len(single.projection.dependencies_of(single_root)),
            1,
        )
        self.assertEqual(
            len(repeated.projection.dependencies_of(repeated_root)),
            2,
        )

    def test_deterministic_operator_kinds_do_not_affect_graph_hashes(self) -> None:
        add_source = Normal(0, 1)
        add_root = Normal(add_source + 1, 1)
        multiply_source = Normal(0, 1)
        multiply_root = Normal(multiply_source * 2, 1)

        add_hashes = resolve_stochastic_hashes(add_root)
        multiply_hashes = resolve_stochastic_hashes(multiply_root)

        self.assertEqual(
            add_hashes.for_node(add_source).final,
            multiply_hashes.for_node(multiply_source).final,
        )
        self.assertEqual(
            add_hashes.for_node(add_root).final,
            multiply_hashes.for_node(multiply_root).final,
        )

    def test_rng_label_is_mixed_after_the_graph_hash(self) -> None:
        left = Normal(0, 1, rng_label="left")
        right = Normal(0, 1, rng_label="right")
        left_hashes = resolve_stochastic_hashes(left)
        right_hashes = resolve_stochastic_hashes(right)

        self.assertEqual(
            left_hashes.for_node(left).final,
            right_hashes.for_node(right).final,
        )
        self.assertNotEqual(
            left_hashes.node_entropy_for(left),
            right_hashes.node_entropy_for(right),
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
        self.assertFalse(hasattr(expr, "value"))
        self.assertTrue(hasattr(checkpoint, "value"))
        with self.assertRaises(TypeError):
            _ = cast(Any, checkpoint) + 1

    def test_checkpoint_value_requires_a_constant_root(self) -> None:
        with sampling_phase("draw"):
            expr = Normal(0, 1)

        checkpoint = expr.materialize(seed=1, phases=())

        self.assertFalse(checkpoint.is_fully_materialized)
        with self.assertRaisesRegex(
            UnrealizedGraphError,
            "checkpoint still contains unrealized stochastic nodes",
        ):
            checkpoint.value()

    def test_completed_checkpoint_retains_source_graph_plate_sizes(self) -> None:
        expr = Normal(0, 1).add_plates("col", "row").mean("col")
        checkpoint = expr.materialize(
            seed=1,
            plate_sizes={"col": 3, "row": 2},
        )

        value = checkpoint.realize(
            seed=2,
            plate_sizes={"col": 3, "row": 2},
        )

        self.assertEqual(value.layout.plates, ("row",))
        self.assertEqual(value.shape, (2,))


if __name__ == "__main__":
    unittest.main()
