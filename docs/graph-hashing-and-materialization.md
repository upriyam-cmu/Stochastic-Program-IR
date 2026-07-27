# Graph Hashing and Materialization Architecture

This document explains the two least obvious v0.1 mechanisms: stochastic-only
graph hashing and immutable partial materialization. It is descriptive of the
current implementation; the normative product contract remains
[`specs/specification-v0.1.md`](../specs/specification-v0.1.md).

## 1. Responsibilities

The components have deliberately narrow roles:

| Component | Responsibility |
| --- | --- |
| `RandomVariable.dependencies` | Return a complete, name-sorted dependency tuple |
| `rewrite_dependencies(mapping)` | Rebuild the same node type with exactly those rewritten dependencies |
| `StochasticProjection` | Contract deterministic nodes into a multigraph between distribution nodes |
| `ResolvedGraphHashes` | Record auditable dependency, consumer, enumeration, and final hashes |
| `NodeEntropy` | Mix one final graph hash with the optional user label |
| materialization pass | Bind run seeds, sample ready nodes, and fold deterministic regions |
| `SamplingCheckpoint` | Opaque immutable root around the rewritten, entropy-resolved graph |

No external graph manager owns nodes. Traversal state, identity memoization, and
rewrite bookkeeping are local to each pass.

## 2. Why the hash graph contains only stochastic nodes

Only distribution nodes consume RNG state. The hashing pass therefore preserves
the relationships that distinguish stochastic draws while contracting
deterministic computation between them.

For example:

```text
A ──┐
    ├─ (A + B) - (C * D) ─► X
B ──┤
C ──┤
D ──┘
```

If `A`, `B`, `C`, `D`, and `X` are distributions, the projection gives `X`
four stochastic input occurrences. The deterministic operator kinds are not
encoded, but each occurrence retains:

- the dependency parameter of `X` through which it arrives; and
- its left-to-right ordinal within that deterministic frontier.

This is intentional. The hash is a semi-stable stochastic address, not a full
serialization of the computation DAG. Structural equality remains responsible
for comparing deterministic computation.

## 3. The projected multigraph

Projection walks from the requested expression root toward its dependencies.
For every named dependency of a distribution consumer, it finds the nearest
upstream distributions through deterministic nodes. Each occurrence becomes an
edge:

```text
(source distribution, consumer distribution, parameter name, ordinal)
```

Edges form a multiset rather than a set. If a deterministic parameter consumes
the same source twice, two edges survive:

```python
source = Normal(0, 1)
single = Normal(source, 1)            # one projected edge
repeated = Normal(source + source, 1) # two projected edges
```

The two graphs therefore derive different hashes even though both projected
edges point to the same source object. Reusing the same node still represents
one sampled value; edge multiplicity records that value's repeated use.

The nearest distributions on the root's stochastic frontier receive synthetic
`"__output__"` consumer edges. This gives root-facing usage local influence
without hashing every descendant path or coupling a reused subgraph to an
arbitrary deeper expression root.

## 4. Multi-stage hash derivation

For each distribution node `v`, the pass computes four values.

### 4.1 Dependency hash `D(v)`

`D(v)` includes:

- the v0.1 hash-scheme domain;
- the distribution implementation type;
- for each projected stochastic input, in canonical dependency-name order:
  - parameter name;
  - occurrence ordinal;
  - the source distribution's dependency hash.

This recursively captures the upstream stochastic structure.

### 4.2 Consumer hash `C(v)`

For every direct projected consumer edge from `v`, the pass hashes:

- the consumer's dependency hash, or the synthetic output marker;
- the consumer parameter name;
- the occurrence ordinal.

Those edge hashes are sorted and hashed as a multiset. Thus the result is
independent of incidental traversal order, while duplicate consumption remains
visible. Deeper descendants beyond the direct stochastic consumer are not
included.

### 4.3 Enumeration `E(v)`

Separately allocated nodes can have identical `D` and `C` context. Canonical
projection traversal assigns each such node a distinct ordinal in its `(D, C)`
bucket. Aliases are visited only once, so only genuinely distinct distribution
nodes receive different enumeration values.

Consequently, two distinct symmetric nodes cannot receive the same final hash
merely because their local structure matches. Sharing requires reusing the same
node object in the source graph.

### 4.4 Final graph hash `G(v)`

```text
G(v) = H_v0.1(D(v), C(v), E(v))
```

Object identities are lookup keys for a single in-process pass only. They are
never digest inputs.

## 5. Deliberate hash exclusions

The graph-structure hash excludes:

- deterministic operator kinds and constants;
- plate layouts and plate operations;
- phase names and barriers;
- user-facing RNG labels.

These exclusions keep stochastic addresses stable across changes that do not
alter the projected stochastic relationships. They are not claims of full graph
equivalence. Structural equality separately compares deterministic nodes,
constants, and plate operations; checkpoint stochastic equality separately
compares phase and RNG-resolution state.

The user-facing `rng_label` is mixed after structural hashing:

```text
node_entropy = H_v0.1(G(v), label-or-no-label)
```

It supplements the graph-derived address and never overrides it. Giving two
distinct nodes the same label therefore cannot couple their draws.

## 6. Materialization seed lifetime

Each call to `materialize(...)` selects exactly one run seed:

- the supplied `seed`; or
- one fresh random 64-bit seed when omitted.

When a distribution is phase-eligible, the pass immediately binds:

```text
sampling_seed = H_v0.1(run_seed, node_entropy)
```

and removes that distribution's phase barrier in the rewritten node. This
happens even if a stochastic dependency is still unresolved.

Suppose `"observation"` is enabled before its `"latent"` dependency:

```text
call 1, seed A:
    observation gets sampling seed H(A, observation entropy)
    observation remains unsampled because latent is blocked

call 2, seed B:
    latent gets sampling seed H(B, latent entropy) and samples
    observation becomes ready and samples with its already stored seed from A
```

This makes phase permission monotonic and prevents a later pass from silently
changing randomness that was fixed when the phase was enabled.

## 7. Immutable dependency rewriting

Materialization must reconstruct arbitrary node types without assuming that a
dependency name is also a dataclass field name. Every node therefore owns:

```python
node.rewrite_dependencies(
    {"dependency_name": rewritten_dependency, ...}
)
```

The public wrapper validates that:

- the mapping contains every existing dependency exactly once;
- every replacement is a `RandomVariable`;
- the hook returns the same concrete node type;
- the rewritten node exposes the same dependency slots.

The protected node hook decides how its constructor fields correspond to those
slots. This canonicalizes the current built-in rewrite pass without yet
promising general custom-node execution support.

## 8. Concrete folding and checkpoints

After recursively rewriting dependencies:

- a ready eligible distribution samples and becomes a `Constant`;
- a deterministic node whose dependencies are constants evaluates through its
  `_evaluate_concrete(...)` hook and becomes a `Constant`;
- an unresolved node is rebuilt around its rewritten dependencies.

`AddPlates` is handled as stochastic lifting: its plate layout propagates toward
the stochastic frontier and into concrete broadcasting. Reduction nodes protect
against incorrectly lifting an identically named external plate through an
existing reduction.

The result is wrapped in `SamplingCheckpoint`, which exposes:

- `materialize(...)` for another immutable rewrite;
- `value()` for extraction only when the root is already a constant;
- `realize(...)` for full remaining materialization followed by extraction;
- checkpoint-only stochastic equality.

Raw expressions do not expose `value()`. They expose `realize()`, which directly
evaluates a fully deterministic tree when `has_value` is true and otherwise
creates a checkpoint through materialization.

## 9. Deferred decisions

The following are intentionally not generalized in this slice:

- public custom-node registration or execution guarantees;
- detaching an internal rewritten graph from a checkpoint;
- lowering parameterized distributions to standardized base distributions;
- alternate numeric backends;
- hashing plates, phases, or deterministic operations;
- public standalone RNG-resolution passes.

These can be reconsidered only after the current authoring, equality, and staged
materialization behavior has been validated.
