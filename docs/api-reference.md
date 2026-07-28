# Public API reference

Only the symbols documented here are supported as v0.1 public API.

## Constructors and transforms

```{autofunction} stoch_ir.constant
```

```{autofunction} stoch_ir.Normal
```

```{autofunction} stoch_ir.Uniform
```

```{autofunction} stoch_ir.Bernoulli
```

```{autofunction} stoch_ir.exp
```

```{autofunction} stoch_ir.log
```

```{autofunction} stoch_ir.softplus
```

```{autofunction} stoch_ir.sampling_phase
```

## Expressions and values

```{autoclass} stoch_ir.RandomVariable
:members: dependencies, plates, pending_phases, has_value, value_meta, add_plates, check_plates, reduce_plates, mean, sum, max, min, prod, logsumexp, exp, log, softplus, abs, materialize, realize, structurally_equal
```

```{autoclass} stoch_ir.SamplingCheckpoint
:members: pending_phases, is_fully_materialized, structurally_equal, stochastically_equal, materialize, value, realize
```

```{autoclass} stoch_ir.ConcreteValue
:members: data, plates, shape, meta, dtype, support
```

```{autoclass} stoch_ir.DataType
:members:
```

```{autoclass} stoch_ir.ValueSupport
:members:
```

```{autoclass} stoch_ir.ValueMeta
:members: dtype, support
```

## Reductions

```{automodule} stoch_ir.reductions
:members:
```

## Errors

```{automodule} stoch_ir.errors
:members:
```
