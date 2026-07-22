# v0.1 Implementation Plan

This plan begins after approval of the API stubs and the normative specification. It intentionally contains no placeholder integrations or speculative extension points: every listed component is needed for the v0.1 acceptance behavior.

## Deliverable 1: immutable graph core

Build the six IR node families and scalar-to-constant coercion.

Required work:

- immutable expression/node representation;
- child traversal and DAG sharing;
- operator overloads for `+`, `-`, `*`, and `/`;
- unary nodes for `exp`, `log`, and `softplus`;
- distribution nodes for `Normal`, `Uniform`, and `Bernoulli`;
- exact, alias-preserving structural equality;
- cycle and duplicate explicit RNG-name validation.

Acceptance checks:

- constructing expressions performs no sampling;
- attempts to mutate nodes fail;
- equivalent separately built DAGs compare equal;
- shared-node and duplicate-node graphs compare unequal;
- phases and RNG names participate in equality.

## Deliverable 2: plate algebra and value alignment

Implement named plate inference, contracts, reductions, and the minimal concrete alignment required by the backend boundary.

Required work:

- `plates` and deterministic `plate_order` inference;
- `add_plates(*plates, expect=None)`;
- `check_plates(*plates)`;
- `reduce_plates` plus the six convenience reduction methods;
- strictly positive materialization-time plate sizes;
- alignment of concrete values by named plate order;
- independent stochastic-frontier lifting through `AddPlates`, without resampling stochastic parameters;
- the public plate error hierarchy.

Acceptance checks:

- duplicate plate introduction fails;
- an incorrect `expect` fails before graph transformation;
- `check_plates` returns the same object;
- reducing a missing plate fails;
- `Normal(row_value, col_value)` aligns to logical `(row, col)`;
- adding a plate to a distribution creates independent draws, not repeated values;
- adding a plate to a conditional distribution does not implicitly resample its stochastic parameters;
- adding a plate to a constant broadcasts the fixed value.

## Deliverable 3: phase annotation

Implement distribution-only sampling phases.

Required work:

- a context-local current phase;
- nested `sampling_phase` restoration;
- phase capture by distribution constructors only;
- recursive `pending_phases` introspection;
- explicit representation of unphased nodes as phase `None`.

Acceptance checks:

- deterministic nodes created inside a phase do not acquire a phase;
- nested contexts assign the innermost phase and restore the outer phase;
- pending phases reflect only reachable, unmaterialized distributions.

## Deliverable 4: RNG derivation and NumPy backend

Implement the one required backend and deterministic graph-to-backend request path.

Required work:

- stable seed normalization;
- canonical graph addresses for unnamed distributions;
- explicit `rng_name` addressing and duplicate detection;
- per-plate-index key derivation;
- `SampleRequest`, `RNGKey`, and `SamplingBackend` contract;
- `NumPyBackend` support for the three v0.1 distributions;
- concrete deterministic arithmetic, transforms, and reductions.

Acceptance checks:

- the same graph and seed reproduce exactly;
- different seeds alter pending stochastic nodes;
- unrelated graph construction does not perturb a result;
- explicit RNG names preserve draws across non-semantic graph refactors;
- NumPy receives only concrete parameters, shape, and an opaque key.

## Deliverable 5: immutable partial materialization

Implement staged graph rewriting and value extraction.

Required work:

- `materialize(phases=..., seed=..., backend=..., plate_sizes=...)`;
- monotonic enabled-phase annotations;
- dependency-blocked enabled nodes that resume on a later pass;
- distribution-to-constant replacement;
- eager constant folding for deterministic downstream nodes;
- unchanged subgraph sharing;
- `realize`, `value`, and `assert_fully_realized`;
- materialization error reporting.

Acceptance checks:

- the source graph remains unchanged;
- materializing one phase leaves other distributions symbolic;
- enabling a downstream phase before its dependency retains the permission;
- later enabling the dependency finishes all newly ready enabled work;
- two branches share embedded early-phase constants;
- later-phase seeds can vary independently;
- value extraction fails while stochastic nodes remain.

## Deliverable 6: product validation suite and examples

Validate the authoring experience rather than only line coverage.

Required work:

- one small hierarchical model example;
- one row/column plate-alignment example;
- one fixed-latent/repeated-observation example;
- expected-graph fixtures suitable for testing generated code;
- error-message snapshots for common plate mistakes;
- README examples converted to executable tests.

The release candidate is acceptable only if each example is shorter or materially more explicit about stochastic structure than its equivalent imperative NumPy program.

## Test matrix

| Concern | Required cases |
| --- | --- |
| Equality | equal DAGs, different args, different phase, different RNG name, shared vs duplicated node |
| Plates | add, duplicate add, exact check, failed expectation, ordered union, each reduction |
| Alignment | scalar/scalar, row/scalar, row/col, reduced row/col |
| Phases | named, unphased, nested context, blocked enabled phase, enable all |
| RNG | same seed, different seed, unrelated construction, explicit stable name |
| Materialization | none eligible, some eligible, all eligible, immutable source, branch reuse |
| Errors | invalid plate, missing size, duplicate RNG name, backend failure, premature value extraction |

## Release deliverables

The v0.1 release must contain:

- the public interfaces currently described by the `.pyi` files;
- one NumPy implementation of `SamplingBackend`;
- full implementations of the six node families;
- plate-aware deterministic evaluation and the six reductions;
- phase contexts and partial materialization;
- structural equality and documented error messages;
- tests for the matrix above;
- the three end-to-end examples;
- package metadata and generated API documentation.

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

## Decision gates

Before implementation begins, confirm:

1. package and import name;
2. whether `rng_name` remains optional or becomes required for distribution nodes;
3. whether exact constant equality must support only NumPy values in v0.1;
4. whether `plate_order` is public API or documented runtime metadata;
5. whether unphased distributions should remain expressible or automatically use a named default phase.

These are contract decisions, not invitations to widen scope. All other post-v0.1 ideas should be recorded separately and must not block the MVP.
