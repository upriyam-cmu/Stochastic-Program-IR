# Concrete values and metadata

{class}`stoch_ir.ConcreteValue` is the public result wrapper. It exposes:

- `data`: an immutable NumPy array;
- `plates`: the canonical tuple matching the array axes;
- `shape`: the NumPy shape;
- `meta`: combined value metadata;
- `dtype`: Boolean, integer, or floating metadata; and
- `support`: conservative range metadata.

## Constants

```python
import numpy as np

from stoch_ir import DataType, constant

offset = constant(
    np.array([1, 2], dtype=np.int8),
    plates=("group",),
    dtype=DataType.FLOAT,
)
```

`ValueMeta.dtype` is authoritative. Values are stored as `np.bool_`,
`np.int64`, or `np.float64`, and narrow support declarations are validated
after coercion.

Multidimensional arrays require one named plate per axis. Complex, object, and
string arrays are rejected. The declared plate order corresponds to the input
array's axis order; the boundary transposes data into canonical lexicographic
plate order. A bare string such as `plates="group"` is one plate, not an
iterable of characters.
