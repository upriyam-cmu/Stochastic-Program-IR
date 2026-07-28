# Named plates and explicit contractions

Plates name independent replication dimensions. They describe stochastic
structure, not merely NumPy shape.

## Introduce and validate plates

```python
from stoch_ir import Normal

weights = Normal(0.0, 1.0).add_plates("feature")
batched = Normal(weights, 1.0).add_plates(
    "batch",
    expect=("feature",),
)
batched.check_plates("batch", "feature")
```

`expect` checks the complete input plate set before introducing the requested
new plate. This makes accidental replication visible near its source.

`expr.plates` returns a canonical lexicographic tuple. Contract methods accept
the expected names in any order.

## Named alignment

Binary operations align operands by plate name and take the union of their
plates. They do not depend on the operand's positional axis order.

## Matrix products without hidden contractions

```python
from stoch_ir import Normal

left = Normal(0.0, 1.0).add_plates("row", "inner")
right = Normal(0.0, 1.0).add_plates("inner", "col")

product = (
    (left * right)
    .sum("inner")
    .check_plates("row", "col")
)
```

This is the named-plate equivalent of a matrix product. Multiplication aligns
`"inner"` by name; `sum("inner")` explicitly identifies the contracted plate.
An implicit `@` operator would hide that choice, so it is deliberately omitted
from v0.1.

## Reductions

The convenience methods `mean`, `sum`, `max`, `min`, `prod`, and `logsumexp`
all delegate to `reduce_plates`. The general form accepts one of the immutable
objects in {mod}`stoch_ir.reductions`.
