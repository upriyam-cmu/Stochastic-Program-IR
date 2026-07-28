# Contributor architecture

The public random-variable algebra is backed by immutable expression nodes,
canonical named-plate metadata, and an immutable materialization rewrite.

These pages describe internals for contributors; they are not extension
contracts:

- [Graph hashing and materialization](graph-hashing-and-materialization.md)
- [v0.1 implementation plan](implementation-plan-v0.1.md)

Concrete node classes, dependency reconstruction, operation implementations,
plate layouts, and stochastic hash passes remain internal in v0.1.

The normative product specification remains in `specs/specification-v0.1.md`
in the source repository.
