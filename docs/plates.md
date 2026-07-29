# Named plates and explicit contractions

Plates name array axes. A distribution makes conditionally independent draws
over its output plates, while `add_plates` broadcasts an existing value without
resampling it.

## Introduce and validate plates

```python
from stoch_ir import Normal

weights = Normal(0.0, 1.0, plates="feature")
batched = Normal(
    weights,
    1.0,
    plates=("feature", "batch"),
)
batched.check_plates("batch", "feature")
```

The `plates=` argument is the distribution's complete output layout. Every
parameter plate must occur in it. A parameter is broadcast over output plates
it does not have; the distribution still makes a new draw at every output
coordinate.

When `plates` is omitted, a distribution uses the canonical union of its
parameter plates, matching ordinary vectorized NumPy sampling.

`expr.plates` returns a canonical lexicographic tuple. Contract methods accept
the expected names in any order.

## Named alignment

Binary operations align operands by plate name and take the union of their
plates. They do not depend on the operand's positional axis order.

## Broadcast existing values

```python
source = Normal(0.0, 1.0)
broadcast = source.add_plates("batch", expect=())
independent = Normal(0.0, 1.0, plates="batch")
```

`broadcast` contains one sampled value repeated over `"batch"`. `independent`
contains one draw for each batch coordinate. `expect` checks the complete input
plate set before broadcasting and creates no validation node.

## Matrix products without hidden contractions

```python
from stoch_ir import Normal

left = Normal(0.0, 1.0, plates=("row", "inner"))
right = Normal(0.0, 1.0, plates=("inner", "col"))

product = (left * right).sum("inner").check_plates("row", "col")
```

This is the named-plate equivalent of a matrix product. Multiplication aligns
`"inner"` by name; `sum("inner")` explicitly identifies the contracted plate.
An implicit `@` operator would hide that choice, so it is deliberately omitted
from v0.1.

## Reductions

The convenience methods `mean`, `sum`, `max`, `min`, `prod`, and `logsumexp`
all delegate to `reduce_plates`. The general form accepts one of the immutable
objects in {mod}`stoch_ir.reductions`.
