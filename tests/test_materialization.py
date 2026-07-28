import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np

from stoch_ir import (
    Normal,
    RandomVariable,
    SamplingCheckpoint,
    sampling_phase,
)
from stoch_ir.errors import (
    MissingPlateSizeError,
    PhaseError,
    PlateSizeMismatchError,
    UnrealizedGraphError,
    UnresolvedRandomnessError,
)
from stoch_ir.expr.hashing import (
    resolve_stochastic_hashes,
)
from stoch_ir.expr.nodes.distr import Gaussian, RandomDistributionNode
from stoch_ir.rng import NodeEntropy


class MaterializationTests(unittest.TestCase):
    def test_plate_sampling_is_repeatable_for_seed_and_label(self) -> None:
        expr = Normal(
            0,
            1,
            plates="row",
            rng_label="weights",
        )

        first = expr.realize(seed=7, plate_sizes={"row": 4})
        second = expr.realize(seed=7, plate_sizes={"row": 4})

        np.testing.assert_array_equal(first.data, second.data)
        self.assertEqual(first.layout.plates, ("row",))

    def test_partial_materialization_reuses_earlier_phase(self) -> None:
        with sampling_phase("latent"):
            latent = Normal(0, 1)
        with sampling_phase("observation"):
            observation = Normal(latent, 1, plates="row")

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

    def test_distribution_output_plates_affect_graph_hashes(self) -> None:
        row = Normal(0, 1, plates="row")
        col = Normal(0, 1, plates="col")

        row_hashes = resolve_stochastic_hashes(row)
        col_hashes = resolve_stochastic_hashes(col)

        self.assertNotEqual(
            row_hashes.for_node(row).final,
            col_hashes.for_node(col).final,
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
        expr = Normal(0, 1, plates=("col", "row")).mean("col")
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

    def test_materialization_requires_valid_sizes_for_every_plate(self) -> None:
        expr = Normal(0, 1, plates="row")

        with self.assertRaises(MissingPlateSizeError):
            expr.realize(seed=1)
        for invalid in (0, -1, True, 1.5):
            with (
                self.subTest(size=invalid),
                self.assertRaises(PlateSizeMismatchError),
            ):
                expr.realize(
                    seed=1,
                    plate_sizes=cast(Any, {"row": invalid}),
                )

    def test_checkpoint_sizes_are_fixed(self) -> None:
        checkpoint = Normal(0, 1, plates="row").materialize(
            seed=1,
            plate_sizes={"row": 2},
            phases=(),
        )

        with self.assertRaises(PlateSizeMismatchError):
            checkpoint.materialize(plate_sizes={})
        with self.assertRaises(PlateSizeMismatchError):
            checkpoint.materialize(plate_sizes={"row": 3})
        with self.assertRaises(PlateSizeMismatchError):
            checkpoint.materialize(plate_sizes={"row": True})

    def test_deterministic_graph_is_evaluated_without_checkpoint(self) -> None:
        source = cast(Gaussian, Normal(0, 1))
        deterministic = source.mu + source.mu

        value = deterministic.realize()

        self.assertEqual(value.data, 0)

    def test_unresolved_distribution_cannot_be_wrapped_as_checkpoint(self) -> None:
        with self.assertRaises(UnresolvedRandomnessError):
            SamplingCheckpoint._wrap(Normal(0, 1), {})

    def test_entropy_cannot_be_replaced_or_seeded_too_early(self) -> None:
        node = cast(RandomDistributionNode, Normal(0, 1))
        with self.assertRaises(UnresolvedRandomnessError):
            node.bind_sampling_seed(1)
        stamped = node.with_node_entropy(NodeEntropy(b"first"))
        with self.assertRaises(UnresolvedRandomnessError):
            stamped.with_node_entropy(NodeEntropy(b"second"))
        bound = stamped.bind_sampling_seed(1)
        self.assertEqual(
            bound.bind_sampling_seed(2)._sampling_seed,
            bound._sampling_seed,
        )

    def test_stochastic_equality_checks_seeds_sizes_and_structure(self) -> None:
        left = Normal(0, 1, plates="row").materialize(
            seed=1,
            plate_sizes={"row": 2},
            phases=(),
        )
        same = Normal(0, 1, plates="row").materialize(
            seed=1,
            plate_sizes={"row": 2},
            phases=(),
        )
        different_seed = Normal(0, 1, plates="row").materialize(
            seed=2,
            plate_sizes={"row": 2},
            phases=(),
        )
        different_size = Normal(0, 1, plates="row").materialize(
            seed=1,
            plate_sizes={"row": 3},
            phases=(),
        )
        different_graph = Normal(1, 1, plates="row").materialize(
            seed=1,
            plate_sizes={"row": 2},
            phases=(),
        )

        self.assertTrue(left.stochastically_equal(same))
        self.assertFalse(left.stochastically_equal(different_seed))
        self.assertFalse(left.stochastically_equal(different_size))
        self.assertFalse(left.stochastically_equal(different_graph))
        self.assertFalse(left.stochastically_equal(cast(Any, object())))
        self.assertFalse(left.structurally_equal(cast(Any, object())))

    def test_reduced_plate_can_be_reintroduced_by_broadcasting(self) -> None:
        expr = Normal(0, 1, plates="row").mean("row").add_plates("row")

        value = expr.realize(seed=1, plate_sizes={"row": 3})

        self.assertEqual(value.plates, ("row",))
        self.assertEqual(value.shape, (3,))
        np.testing.assert_array_equal(value.data, np.full(3, value.data[0]))

    def test_add_plates_broadcasts_one_sample(self) -> None:
        expr = Normal(0, 1).add_plates("row")

        value = expr.realize(seed=1, plate_sizes={"row": 4})

        np.testing.assert_array_equal(value.data, np.full(4, value.data[0]))

    def test_distribution_plates_create_independent_samples(self) -> None:
        expr = Normal(0, 1, plates="row")

        value = expr.realize(seed=1, plate_sizes={"row": 4})

        self.assertGreater(np.unique(value.data).size, 1)

    def test_aliases_broadcast_over_different_plates_without_rng_coupling(
        self,
    ) -> None:
        source = Normal(0, 1)
        expr = source.add_plates("row") - source.add_plates("col")

        value = expr.realize(
            seed=1,
            plate_sizes={"row": 3, "col": 4},
        )

        np.testing.assert_array_equal(value.data, np.zeros((4, 3)))

    def test_invalid_materialization_phase_does_not_mutate_source(self) -> None:
        with sampling_phase("draw"):
            expr = Normal(0, 1)

        with self.assertRaises(PhaseError):
            expr.materialize(seed=1, phases=("",))
        self.assertEqual(expr.pending_phases, frozenset({"draw"}))


def test_v01_hash_digest_fixture() -> None:
    first = Normal(0, 1)
    second = Normal(first + first, 1)
    hashes = resolve_stochastic_hashes(second)

    assert hashes.for_node(first).final.hex() == "8374705384dc9f224cb5000969433b72"
    assert hashes.for_node(second).final.hex() == "b4230971a6bddec783123c532eac5a84"


def test_hash_digest_is_stable_across_python_hash_seeds() -> None:
    root = Path(__file__).resolve().parents[1]
    code = (
        "from stoch_ir import Normal;"
        "from stoch_ir.expr.hashing import "
        "resolve_stochastic_hashes;"
        "a=Normal(0,1);b=Normal(a+a,1);"
        "h=resolve_stochastic_hashes(b);"
        "print(h.for_node(a).final.hex(),h.for_node(b).final.hex())"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=root,
                env=environment,
                text=True,
            ).strip()
        )

    assert outputs[0] == outputs[1]


if __name__ == "__main__":
    unittest.main()
