# Stochastic Programming Library v0.1 Specification

This document is the normative design contract for v0.1. The words **must**, **should**, and **may** describe required, recommended, and optional behavior respectively.

## 1. Product objective

v0.1 must test whether a small graph-first API makes forward-sampling programs easier to author and verify than imperative RNG code, especially when programs are generated or modified with agentic assistance.

The product must make these relationships salient in source code and inspectable in the resulting graph:

- stochastic dependencies;
- shared versus independent samples;
- plate introduction and reduction;
- exact plate expectations at important boundaries;
- sampling-phase membership;
- partial versus complete materialization.

The product is not a probabilistic inference system. It performs forward sampling only.

## 2. Design principles

### 2.1 One immutable representation

Every program is an immutable expression DAG. Graph construction, validation, and materialization return expressions and never mutate an existing expression.

A partially materialized program is represented by the same expression API as an untouched program. Successfully sampled distribution nodes are replaced by constant nodes in a new DAG.

### 2.2 Exact stochastic intent

The API must not silently infer whether the caller intended to introduce, preserve, or remove a plate:

- `add_plates` explicitly introduces independent replication;
- `check_plates` explicitly validates the complete current plate set;
- `reduce_plates` explicitly removes plates using a named reduction;
- `add_plates(expect=...)` validates the complete pre-transformation plate set before adding the requested plates.

### 2.3 Orthogonal plates and phases

Plates describe **where independent multiplicity exists**. Phases describe **when a distribution node may be sampled**. Neither changes the semantics of the other.

### 2.4 Backend isolation

The graph layer owns graph traversal, dependency readiness, plate alignment, phase eligibility, RNG-key derivation, and graph rewriting. A backend only samples a concrete distribution request using an opaque RNG key.

## 3. Terms

### Expression

An immutable node in a symbolic DAG. Every public stochastic or deterministic operation returns an `Expr`.

### Plate

A string identifier for an independent replication axis, such as `"batch"`, `"row"`, or `"particle"`.

Plate names do not contain sizes. Concrete sizes are supplied during materialization so the same graph can be realized at different extents.

### Phase

A string attached to a distribution node at construction time. A distribution constructed outside `sampling_phase(...)` has phase `None`.

### Materialization

An immutable graph rewrite that samples eligible distribution nodes and replaces them with constants.

### Realization

Full materialization followed by extraction of the root concrete value.

## 4. Public expression model

Every expression must expose:

```python
expr.plates: frozenset[str]
expr.plate_order: tuple[str, ...]
expr.pending_phases: frozenset[str | None]
expr.enabled_phases: frozenset[str | None]
expr.is_fully_realized: bool
```

`plates` is the semantic set used by validation. `plate_order` is the deterministic axis order used to align concrete values. v0.1 orders plates by first appearance during expression construction:

- distribution parameters are visited in their declared parameter order;
- binary operators visit the left operand before the right operand;
- `add_plates` appends newly added plates in call order;
- reductions preserve the order of remaining plates.

Plate order is not part of `check_plates`, which compares sets. It is part of materialized value metadata and exact graph equality.

### 4.1 Inputs and constants

Python scalar inputs are automatically represented as plate-free constants. A public `Constant(value, plates=...)` constructor must allow callers to inject an existing backend value and describe its named axes.

For a non-scalar constant, its number and order of declared plates must agree with its runtime rank when the selected backend exposes rank. This validation may occur at construction or first materialization.

### 4.2 Core node families

The v0.1 IR contains exactly these node families:

1. `Constant`;
2. `Distribution`;
3. `UnaryOperator`;
4. `BinaryOperator`;
5. `AddPlates`;
6. `Reduction`.

Validation calls do not create nodes.

## 5. Distributions and transforms

v0.1 must specify these distributions:

```python
Normal(mu, sigma, *, rng_name=None)
Uniform(low, high, *, rng_name=None)
Bernoulli(p, *, rng_name=None)
```

Parameters accept expressions or scalar constants. A distribution's plates are the ordered union of its parameter plates plus any plates introduced around the expression through `add_plates`.

`rng_name` is an optional stable, human-readable RNG address within the reachable graph. Duplicate explicit RNG names in one reachable graph are invalid. When omitted, the graph engine derives a deterministic address from exact DAG topology. Explicit names are recommended at semantic boundaries and in agent-generated programs.

v0.1 must specify:

- binary `+`, `-`, `*`, and `/`;
- unary `exp(expr)`, `log(expr)`, and `softplus(expr)`.

Deterministic operators use the ordered union of operand plates and must align concrete operands by plate name before applying a backend-neutral numeric operation.

## 6. Plate operations

### 6.1 `add_plates`

```python
expr.add_plates(*plates, expect=None) -> Expr
```

`add_plates` adds exactly the requested new independent replication plates.

Preconditions:

- every plate must be a non-empty string;
- no requested plate may occur more than once;
- no requested plate may already exist on `expr`;
- when `expect` is provided, `expr.plates` must exactly equal `frozenset(expect)` before any plate is added.

`expect` is validation sugar only:

```python
expr.add_plates("batch", expect=("feature",))
```

is semantically equivalent to:

```python
expr.check_plates("feature").add_plates("batch")
```

The returned graph contains only an `AddPlates` node, never a validation node.

An `AddPlates` node is a stochastic lifting operation, not a numeric repeat. During materialization, newly introduced plates propagate through deterministic nodes to the stochastic frontier of the child expression. Each distribution reached at that frontier samples independently across the new plates, and propagation stops there: stochastic distribution parameters retain their own plates and are aligned or broadcast rather than implicitly resampled. A deterministic constant with no stochastic frontier is broadcast.

For example, `Normal(mu=x, sigma=1).add_plates("col")` draws a new conditional `Normal` value for each `"col"`, but it does not resample a stochastic `x` separately for each column unless `x` itself was explicitly given that plate.

### 6.2 `check_plates`

```python
expr.check_plates(*plates) -> Expr
```

The method requires exact set equality and returns the same expression object. It must reject duplicates and malformed names. It must not create a graph node or change equality, phase metadata, or evaluation behavior.

### 6.3 `reduce_plates`

```python
expr.reduce_plates(*plates, reduction=Reduction.MEAN) -> Expr
```

Every requested plate must exist and may appear only once. The result removes those plates and preserves the relative order of all remaining plates.

The v0.1 `Reduction` enum contains:

- `MEAN`;
- `SUM`;
- `MAX`;
- `MIN`;
- `PROD`;
- `LOGSUMEXP`.

Arbitrary reduction callables are not supported in v0.1.

Convenience methods (`mean`, `sum`, `max`, `min`, `prod`, and `logsumexp`) must delegate to `reduce_plates` and must not introduce distinct IR node types.

## 7. Plate inference and alignment

Plate inference is local and recursive:

| Node | Plate result |
| --- | --- |
| Constant | declared plates, empty for scalar constants |
| Distribution | ordered union of parameter plates |
| UnaryOperator | child plates |
| BinaryOperator | ordered union of left and right plates |
| AddPlates | child plates followed by newly added plates |
| Reduction | child plates minus reduced plates |

Materialization accepts:

```python
plate_sizes: Mapping[str, int]
```

Every plate needed by a sampled or evaluated non-scalar node must have a strictly positive size. Extra entries may be ignored. Missing required sizes must raise `MissingPlateSizeError` before the backend is called.

Before a deterministic operation or distribution request, the graph engine aligns concrete parameter axes to the expression's `plate_order`. The sampling backend receives only an unnamed output shape and already aligned values; it does not receive plate names.

This named-axis alignment is required for v0.1. General tensor shape inference beyond named plate axes is not.

## 8. Sampling phases

```python
with sampling_phase("latent"):
    x = Normal(...)
```

The context records a default phase only on distribution nodes constructed inside it. Deterministic operators and plate nodes do not store the active phase.

Nested phase contexts use the innermost active phase. Exiting restores the previous phase. Phase names have no ordering or implicit precedence.

An unphased distribution has phase `None`. `None` may be explicitly enabled through `phases=(None,)`; omitting `phases` enables every remaining named and unphased distribution.

## 9. Materialization

The proposed signature is:

```python
expr.materialize(
    *,
    seed,
    backend=None,
    plate_sizes=None,
    phases=None,
) -> Expr
```

`phases=None` means enable every phase still present in the reachable graph. An explicit iterable enables only those phases.

Phase enabling is monotonic for a returned graph. The materializer computes:

```text
enabled = prior enabled phases union requested phases
```

If an enabled distribution is blocked by an unmaterialized dependency, it remains symbolic but its phase remains enabled in the returned graph. A later materialization that clears the dependency must sample that node without requiring its phase to be named again.

The rewrite must:

1. validate the reachable graph and requested plate sizes;
2. recursively rewrite dependencies;
3. eagerly evaluate deterministic nodes whose inputs are constants;
4. sample a distribution only when its phase is enabled and all parameters are concrete;
5. replace each sampled distribution with a constant carrying its value and plate order;
6. preserve untouched subgraphs by reference when possible;
7. return a new root expression carrying any still-relevant enabled-phase annotation.

The original graph must remain unchanged.

### 9.1 `pending_phases`

`pending_phases` is computed from reachable distribution nodes that have not become constants. It includes `None` when unphased distributions remain.

### 9.2 `realize`

```python
expr.realize(*, seed, backend=None, plate_sizes=None) -> Value
```

`realize` enables all remaining phases, fully materializes the graph, verifies that no distribution node remains, and returns the root concrete value. A failure to reach a concrete root raises `UnrealizedGraphError`.

`value()` performs extraction only. It never samples and raises `UnrealizedGraphError` unless the root is already concrete.

### 9.3 Branching and repeatability

A materialized graph embeds completed samples as constants. It can therefore be used as a checkpoint:

```python
fixed = expr.materialize(phases=("latent",), seed=10, plate_sizes=sizes)

a = fixed.realize(seed=20, plate_sizes=sizes)
b = fixed.realize(seed=21, plate_sizes=sizes)
```

Different seeds intentionally resample remaining nodes. Reusing seed `20` must reproduce `a`. Already embedded latent constants are unaffected by the later seeds.

## 10. RNG addressing

The graph layer derives one opaque RNG key per distribution draw from:

- the caller-provided materialization seed;
- the distribution's explicit `rng_name`, when present, otherwise its canonical graph address;
- the indices of all active plates for that draw.

The derivation must be stable across process runs and must not use Python's randomized `hash()`.

Required behavior:

- repeated materialization with the same graph, seed, sizes, and backend is reproducible;
- constructing unrelated graphs does not perturb existing draws;
- graph sharing results in one shared sampled value;
- distinct distribution nodes result in distinct draw addresses, even when their parameters are structurally identical;
- reordering or otherwise changing the reachable graph may change unnamed addresses and is considered a structural change;
- explicit `rng_name` values provide stability across such refactors when their stochastic meaning is intended to remain fixed.

## 11. Backend contract

```python
class SamplingBackend(Protocol):
    def sample(self, request: SampleRequest, *, rng_key: RNGKey) -> Value: ...
```

`SampleRequest` contains:

- a `DistributionKind` enum value;
- concrete, plate-aligned parameters;
- an unnamed output shape.

The backend must not inspect an expression graph and must not receive plate names, phases, or validation metadata.

v0.1 implementation work will provide one `NumPyBackend`. JAX and PyTorch backends are future work and are not stubbed as promised deliverables.

## 12. Structural equality

`Expr.__eq__` and `Expr.structurally_equal` perform exact graph comparison. They must compare:

- node families and operator/distribution kinds;
- ordered arguments and constant values;
- plate names and `plate_order`;
- phase and `rng_name` metadata;
- enabled-phase annotations that can affect future materialization;
- DAG aliasing topology.

Equality must preserve aliasing. A graph that uses the same distribution node twice is not equal to a graph containing two distinct, structurally identical distribution nodes.

Equality is not algebraic: `x + y` need not equal `y + x`, and no simplification such as `x + 0 == x` is performed.

Backend arrays require backend-aware exact value comparison. Approximate numeric comparison is outside structural equality and outside v0.1.

## 13. Errors

The public error hierarchy is:

```text
StochasticProgrammingError
├── GraphValidationError
│   ├── DuplicateRNGNameError
│   └── GraphCycleError
├── PlateError
│   ├── DuplicatePlateError
│   ├── PlateExpectationError
│   ├── UnknownPlateError
│   └── MissingPlateSizeError
├── PhaseError
├── MaterializationError
│   ├── BackendError
│   └── UnrealizedGraphError
```

Errors should report the relevant node, requested operation, expected state, and actual state where applicable.

## 14. Explicit non-goals

v0.1 excludes:

- posterior inference, conditioning, and observations;
- automatic differentiation;
- JIT compilation and graph optimization;
- graph serialization and visualization;
- custom distributions and custom reductions;
- dynamic or data-dependent plates;
- unnamed positional tensor-axis inference;
- advanced indexing, slicing, stacking, concatenation, and matrix multiplication;
- JAX, PyTorch, or distributed backends;
- mutable sampling sessions or global RNG state;
- algebraic graph equivalence.

## 15. Acceptance example

```python
with sampling_phase("latent"):
    x = Normal(0.0, 1.0, rng_name="x").add_plates("row")

with sampling_phase("observation"):
    y = Normal(
        mu=x,
        sigma=1.0,
        rng_name="y",
    ).add_plates("col", expect=("row",))

z = y.mean("col").check_plates("row")

checkpoint = z.materialize(
    phases=("latent",),
    seed=1,
    plate_sizes={"row": 4, "col": 8},
)

assert checkpoint.pending_phases == frozenset({"observation"})

value = checkpoint.realize(
    seed=2,
    plate_sizes={"row": 4, "col": 8},
)
```

The implementation passes the central v0.1 acceptance case when this program produces a concrete value with logical plate `"row"`, preserves the sampled latent values across checkpoint branches, changes observation draws under different later seeds, reproduces them under the same seed, and rejects incorrect plate contracts close to their source.
