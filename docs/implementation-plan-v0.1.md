# v0.1 Implementation Plan

This plan tracks the incremental v0.1 implementation. It intentionally contains
no placeholder integrations or speculative extension points: every listed
component is needed for the acceptance behavior.

## Deliverable 1: immutable graph core

Build the six IR node families and scalar-to-constant coercion.

Required work:

- immutable expression/node representation;
- child traversal and DAG sharing;
- operator overloads for `+`, `-`, `*`, `/`, and `//`;
- unary nodes for `exp`, `log`, `softplus`, and absolute value;
- distribution nodes for `Normal`, `Uniform`, and `Bernoulli`;
- alias-agnostic structural equality;
- cycle validation.

Acceptance checks:

- constructing expressions performs no sampling;
- attempts to mutate nodes fail;
- equivalent separately built DAGs compare equal;
- shared-node and duplicate-node graphs compare structurally equal when their
  computation trees are otherwise equivalent;
- resolved checkpoints distinguish stochastic aliasing, node entropy, and bound
  sampling-seed differences.

## Deliverable 2: plate algebra and value alignment

Implement named plate inference, contracts, reductions, and the minimal concrete alignment required by the backend boundary.

Required work:

- `plates` and canonical lexicographic `plate_layout` inference;
- `add_plates(*plates, expect=None)`;
- `check_plates(*plates)`;
- `reduce_plates` plus the six convenience reduction methods;
- strictly positive materialization-time plate sizes;
- alignment of concrete values by named plate order;
- complete output plate declarations on distribution nodes;
- deterministic broadcasting through `AddPlates`;
- the public plate error hierarchy.

Acceptance checks:

- duplicate plate introduction fails;
- an incorrect `expect` fails before graph transformation;
- `check_plates` returns the same object;
- reducing a missing plate fails;
- `Normal(row_value, col_value)` aligns to logical `(row, col)`;
- distribution output plates create conditionally independent draws;
- distribution output plates must contain every parameter plate;
- adding a plate to any expression broadcasts its existing value.

Concrete-value work:

- expose `constant(value, *, plates=(), dtype=None)`;
- infer supported Boolean, integer, and floating inputs;
- canonicalize storage to `np.bool_`, `np.int64`, or `np.float64`;
- transpose declared input axes into canonical plate order;
- derive and enforce four-state support after coercion;
- reject unsupported kinds and rank/layout mismatches at the boundary.

## Deliverable 3: phase annotation

Implement distribution-only sampling phases.

Required work:

- a context-local current phase;
- nested `sampling_phase` restoration;
- phase capture by distribution constructors only;
- recursive `pending_phases` introspection;
- unphased distribution nodes with no phase barrier.

Acceptance checks:

- deterministic nodes created inside a phase do not acquire a phase;
- nested contexts assign the innermost phase and restore the outer phase;
- pending phases reflect only reachable, unmaterialized distributions.

## Deliverable 4: graph-aware RNG derivation and NumPy execution

Implement deterministic graph-key resolution and concrete NumPy sampling.

Required work:

- stable seed normalization;
- a stochastic-only graph projection;
- dependency hashes, direct-consumer hashes, repeated-use multiplicities, and
  symmetric-node enumeration;
- compressed projected edges for repeated stochastic consumption;
- optional `rng_label` entropy mixed after the graph hash;
- node-entropy resolution coupled to initial materialization;
- one run seed selected per materialization invocation;
- phase-enabled sampling seeds bound immediately, even while dependencies are
  blocked;
- NumPy sampling for the implemented distributions;
- concrete deterministic arithmetic, transforms, and reductions.

`Uniform` must support arbitrary symbolic bounds with unit defaults and
elementwise `low < high` validation. `Bernoulli` must accept symbolic
probabilities, validate `0 <= p <= 1`, and return canonical Boolean values.

Acceptance checks:

- the same graph and seed reproduce exactly;
- different seeds alter pending stochastic nodes;
- unrelated graph construction does not perturb a result;
- labels never override graph structure or couple distinct nodes;
- separately allocated symmetric nodes receive distinct default keys;
- aliases of one node share one resolved entropy identity and sampled value.

## Deliverable 5: immutable partial materialization

Implement staged graph rewriting and value extraction.

Required work:

- `materialize(phases=..., seed=..., plate_sizes=...)`;
- an opaque, non-composable `SamplingCheckpoint`;
- monotonic phase-barrier removal;
- dependency-blocked enabled nodes that resume on a later pass;
- node-owned exact dependency rewriting through a private reconstruction hook;
- distribution-to-constant replacement;
- eager constant folding for deterministic downstream nodes;
- unchanged subgraph sharing;
- checkpoint `materialize` and `realize`;
- checkpoint-only `stochastically_equal`;
- materialization error reporting.

Acceptance checks:

- the source graph remains unchanged;
- materializing one phase leaves other distributions symbolic;
- enabling a downstream phase before its dependency retains the permission;
- later enabling the dependency finishes all newly ready enabled work;
- two branches share embedded early-phase constants;
- later-phase seeds can vary independently;
- value extraction fails while stochastic nodes remain.
- deterministic raw expressions can realize without a checkpoint;
- checkpoint `value()` extracts only an already constant root.

## Deliverable 6: product validation suite and examples

Validate the authoring experience rather than only line coverage.

Required work:

- one small hierarchical model example;
- one row/column plate-alignment example;
- one fixed-latent/repeated-observation example;
- expected-graph fixtures suitable for testing generated code;
- error-message snapshots for common plate mistakes;
- README examples converted to executable tests.
- a public-import-only static typing fixture;
- branch coverage of at least 95%;
- a fixed v0.1 graph-digest fixture and cross-`PYTHONHASHSEED` subprocess check;
- wheel validation proving that `py.typed` and inline annotations ship without
  `.pyi` files.

The release candidate is acceptable only if each example is shorter or materially more explicit about stochastic structure than its equivalent imperative NumPy program.

## Test matrix

| Concern | Required cases |
| --- | --- |
| Equality | equal structures, different args, allocation-order independence, shared vs duplicated stochastic node |
| Plates | add, duplicate add, exact check, failed expectation, ordered union, each reduction |
| Alignment | scalar/scalar, row/scalar, row/col, reduced row/col |
| Phases | named, unphased, nested context, blocked enabled phase, enable all |
| RNG | same seed, different seed, unrelated construction, label mixing, repeated projected edge, symmetric enumeration, blocked-node seed persistence |
| Materialization | none eligible, some eligible, all eligible, immutable source, branch reuse |
| Errors | invalid plate, missing size, numeric sampling failure, premature value extraction |

## Release deliverables

The v0.1 release must contain:

- the typed public runtime interfaces;
- inline annotations and `py.typed`, with no shipped stub files;
- NumPy concrete propagation and distribution sampling;
- full implementations of the six node families;
- plate-aware deterministic evaluation and the six reductions;
- phase contexts and partial materialization;
- structural and checkpoint stochastic equality;
- tests for the matrix above;
- the realistic end-to-end examples;
- package metadata and generated API documentation.
- Python 3.10/latest-supported CI for Ruff, ty, tests, coverage, wheel build,
  wheel inspection, and installed-package smoke tests.

## Deferred work

Do not begin the following during v0.1 unless the core acceptance work proves impossible without it:

- custom extension registries;
- JAX or PyTorch support;
- serialization or visualization;
- optimizers, compilers, tracing, or autodiff;
- inference or observations;
- custom reduction callables;
- rich tensor indexing or dynamic plates;
- public graph visitor or rewrite APIs.

## Locked implementation decisions

- optional `rng_label` supplements graph-derived entropy and never overrides it;
- exact concrete values use NumPy arrays;
- `PlateLayout` is canonicalized lexicographically;
- unphased distributions remain expressible and have no phase barrier;
- a public backend protocol is deferred;
- distribution constructors retain their natural parameters (for example,
  `Normal(mu, sigma)`) rather than lowering to standardized distributions;
- public generic reductions use the immutable singleton objects in
  `stoch_ir.reductions`;
- concrete NumPy storage always matches authoritative `ValueMeta.dtype`;
- custom-node execution support and public graph rewrite APIs are deferred.
