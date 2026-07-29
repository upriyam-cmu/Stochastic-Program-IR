# Alpha stability and intentional omissions

The v0.1 API is an alpha contract. Pin an exact version and expect incompatible
changes between alpha releases.

The project intentionally does not include:

- arbitrary indexing, slicing, concatenation, or positional-axis APIs;
- implicit contractions such as matrix `@`;
- posterior inference, conditioning, or probabilistic-programming effects;
- mutable RNG sessions or global generator state;
- autodiff, JIT compilation, or graph optimization;
- serialization or a stable custom-node format;
- custom distributions or reductions; or
- alternate numeric backends.

These omissions preserve a small algebra whose stochastic semantics are visible
locally in source code. They may be revisited only when a concrete use case can
retain that property.
