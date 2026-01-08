# Migrating from Kingdon to GAlgebra

This guide helps users familiar with the [kingdon](https://github.com/tBuLi/kingdon) library to transition to GAlgebra. GAlgebra now includes many kingdon-inspired features for improved API compatibility.

## Overview

Both kingdon and GAlgebra are Python libraries for Geometric Algebra (GA), but they have different focuses:

- **Kingdon**: Focuses on symbolic optimization, JIT compilation, and multi-backend support (NumPy, PyTorch, SymPy)
- **GAlgebra**: Focuses on symbolic computation with SymPy, geometric calculus, and differential geometry

GAlgebra has added several kingdon-inspired features to provide a more familiar API for kingdon users while maintaining its strong symbolic and calculus capabilities.

## New Features Migrated from Kingdon

### 1. Regressive Product (Meet)

The regressive product (∨), also called the "meet" operation, is now available in GAlgebra.

**Kingdon:**
```python
from kingdon import Algebra

alg = Algebra(3, 0, 1)
plane1 = alg.bivector('p1')
plane2 = alg.bivector('p2')

# Regressive product
meet = plane1 & plane2  # Using & operator
```

**GAlgebra:**
```python
from galgebra.ga import Ga

ga = Ga('e*1|2|3', g=[1, 1, 1])
e1, e2, e3 = ga.mv()

plane1 = e1 ^ e2
plane2 = e1 ^ e3

# Regressive product - now available!
meet = plane1 & plane2  # Using & operator
meet = plane1.rp(plane2)  # Using method
meet = rp(plane1, plane2)  # Using function
```

### 2. Normalized Multivectors

Get a unit multivector by normalizing.

**Kingdon:**
```python
v = alg.vector([1, 2, 3])
v_unit = v.normalized()
```

**GAlgebra:**
```python
v = 1*e1 + 2*e2 + 3*e3
v_unit = v.normalized()  # New method
v_unit = normalized(v)   # New function
```

### 3. Square Root

Compute the square root of multivectors whose square is a scalar.

**Kingdon:**
```python
b = alg.e1 ^ alg.e2
b_sqrt = b.sqrt()
```

**GAlgebra:**
```python
b = e1 ^ e2
b_sqrt = b.sqrt()  # New method
```

### 4. Sandwich Product

The sandwich product (conjugation) is useful for rotations and reflections.

**Kingdon:**
```python
rotor = ...
vector = alg.vector([1, 0, 0])
rotated = rotor.sw(vector)  # or rotor >> vector
```

**GAlgebra:**
```python
rotor = ...
vector = e1
rotated = rotor.sandwich(vector)  # New method
rotated = sandwich(rotor, vector)  # New function
# Equivalent to: rotor * vector * rotor.rev()
```

### 5. API Compatibility Aliases

GAlgebra now provides method aliases that match kingdon's API:

| Kingdon Method | GAlgebra Method | GAlgebra Alias | Description |
|----------------|-----------------|----------------|-------------|
| `.gp(b)` | `a * b` | `a.gp(b)` | Geometric product |
| `.op(b)` | `a ^ b` | `a.op(b)` | Outer/wedge product |
| `.ip(b)` | `a \| b` | `a.ip(b)` | Inner/dot product |
| `.lc(b)` | `a < b` | `a.lc(b)` | Left contraction |
| `.rc(b)` | `a > b` | `a.rc(b)` | Right contraction |
| `.rp(b)` | `a & b` | `a.rp(b)` | Regressive product |
| `.cp(b)` | `a >> b` | `a.cp(b)` | Commutator product |
| `.acp(b)` | `a << b` | `a.acp(b)` | Anti-commutator product |
| `.reverse()` | `a.rev()` or `~a` | `a.reverse()` | Reverse |
| `.involute()` | `a.g_invol()` | `a.involute()` | Grade involution |
| `.conjugate()` | `a.ccon()` | `a.conjugate()` | Clifford conjugate |

**Example:**
```python
# Kingdon style
a = e1 + e2
b = e2 + e3

result = a.gp(b)      # Geometric product
result = a.op(b)      # Outer product
result = a.reverse()  # Reverse
result = a.involute() # Grade involution
```

## API Comparison

### Creating a Geometric Algebra

**Kingdon:**
```python
from kingdon import Algebra

# 3D Projective Geometric Algebra (PGA)
alg = Algebra(3, 0, 1)  # (p, q, r) signature
```

**GAlgebra:**
```python
from galgebra.ga import Ga

# 3D Euclidean space
ga = Ga('e*1|2|3', g=[1, 1, 1])
# Or with signature
ga = Ga('e*1|2|3|4', g=[1, 1, 1, -1])  # Spacetime (3+1)
```

### Creating Multivectors

**Kingdon:**
```python
# Symbolic
v = alg.vector('v')
b = alg.bivector('b')

# Numeric
v = alg.vector([1, 2, 3])
```

**GAlgebra:**
```python
# Get basis vectors
e1, e2, e3 = ga.mv()

# Symbolic
v = ga.mv('v', 'vector')
b = ga.mv('b', 'bivector')

# Numeric
v = 1*e1 + 2*e2 + 3*e3
b = e1^e2 + e2^e3
```

### Operations

**Kingdon:**
```python
# Products
gp = a * b      # Geometric
op = a ^ b      # Outer
ip = a | b      # Inner
rp = a & b      # Regressive

# Unary operations
rev = ~a        # Reverse
dual = a.dual()
inv = a.inv()
```

**GAlgebra:**
```python
# Products (same operators)
gp = a * b      # Geometric
op = a ^ b      # Outer
ip = a | b      # Inner
rp = a & b      # Regressive (NEW!)
lc = a < b      # Left contraction
rc = a > b      # Right contraction

# Unary operations
rev = ~a        # Reverse
dual = a.dual()
inv = a.inv()
```

### Grade Operations

**Kingdon:**
```python
grade_k = mv.grade(k)
```

**GAlgebra:**
```python
grade_k = mv.grade(k)
# Or
grade_k = mv.get_grade(k)
# Or for extraction
grade_k = mv[k]  # Get grade k part
```

## Additional GAlgebra Features

GAlgebra provides additional capabilities beyond kingdon:

### 1. Geometric Calculus

```python
from sympy import symbols

# Define coordinates
x, y, z = symbols('x y z', real=True)
ga = Ga('e*1|2|3', g=[1, 1, 1], coords=(x, y, z))

# Get gradient operator
grad = ga.grad

# Vector field
F = x*y*ga.mv()[0] + z**2*ga.mv()[1]

# Divergence
div_F = grad | F

# Curl
curl_F = -ga.I() * (grad ^ F)

# Laplacian
laplacian_f = (grad * grad) | f
```

### 2. Differential Operators

```python
# Partial derivatives
dF_dx = F.diff(x)

# Covariant derivatives (in curved spaces)
# ...
```

### 3. Linear Transformations

```python
from galgebra.lt import Lt

# Define a linear transformation
L = Lt('L', ga)

# Apply to multivector
result = L(v)
```

### 4. LaTeX Output

```python
from galgebra.printer import Format

Format(Fmode=False, Dmode=True)  # Enable LaTeX mode in Jupyter

# Multivectors will now display beautifully in notebooks
M = ga.mv('M', 'mv')
M.Fmt(3, r'\langle \mathbf{M} \rangle')
```

## Key Differences

### 1. Backend Support

- **Kingdon**: Supports multiple backends (NumPy, PyTorch, SymPy, etc.)
- **GAlgebra**: SymPy-only for symbolic computation

If you need numerical computation with arrays, you may want to use kingdon or combine it with GAlgebra.

### 2. Symbolic Optimization

- **Kingdon**: Automatically optimizes symbolic expressions
- **GAlgebra**: Uses SymPy's simplification; you control when to simplify

```python
# GAlgebra - explicit simplification
from sympy import simplify
result = simplify(complicated_expression)
```

### 3. Coordinate Systems

- **Kingdon**: Basis vectors only
- **GAlgebra**: Full support for arbitrary coordinate systems (spherical, cylindrical, etc.)

```python
# Spherical coordinates
ga_sphere = Ga('e_r e_theta e_phi',
               g=[1, r**2, r**2*sin(theta)**2],
               coords=(r, theta, phi))
```

## Migration Checklist

When migrating from kingdon to GAlgebra:

1. ✅ **Algebra creation**: Update from `Algebra(p, q, r)` to `Ga(...)` with appropriate metric
2. ✅ **Element creation**: Switch from `alg.vector()` to `ga.mv()` or direct construction
3. ✅ **Products**: Most operators are the same (`*`, `^`, `|`)
4. ✅ **Regressive product**: Now available with `&` operator or `.rp()` method
5. ✅ **Method names**: Use new aliases (`.gp()`, `.op()`, etc.) or native GAlgebra methods
6. ✅ **Normalization**: Use `.normalized()` method
7. ✅ **Square root**: Use `.sqrt()` method
8. ✅ **Sandwich product**: Use `.sandwich()` method
9. ⚠️ **Backend**: GAlgebra is SymPy-only; numerical work may require adaptation
10. ⚠️ **Optimization**: Explicitly call `simplify()` when needed

## Example: Complete Migration

**Kingdon code:**
```python
from kingdon import Algebra

alg = Algebra(3, 0, 0)
v1 = alg.vector([1, 2, 3])
v2 = alg.vector([4, 5, 6])

# Normalize
v1_unit = v1.normalized()

# Products
gp = v1.gp(v2)
op = v1.op(v2)

# Rotation
bivector = v1 ^ v2
rotor = bivector.normalized()
rotated = rotor.sw(v1)
```

**GAlgebra equivalent:**
```python
from galgebra.ga import Ga

ga = Ga('e*1|2|3', g=[1, 1, 1])
e1, e2, e3 = ga.mv()

v1 = 1*e1 + 2*e2 + 3*e3
v2 = 4*e1 + 5*e2 + 6*e3

# Normalize (NEW!)
v1_unit = v1.normalized()

# Products
gp = v1.gp(v2)  # or v1 * v2
op = v1.op(v2)  # or v1 ^ v2

# Rotation
bivector = v1 ^ v2
rotor = bivector.normalized()
rotated = rotor.sandwich(v1)  # NEW!
```

## Resources

- **GAlgebra Documentation**: https://galgebra.readthedocs.io/
- **Kingdon Repository**: https://github.com/tBuLi/kingdon
- **GAlgebra Examples**: [examples/](../examples/) directory
- **This Migration Guide Issues**: Report issues at https://github.com/pygae/galgebra/issues

## Getting Help

If you encounter issues during migration:

1. Check the [GAlgebra documentation](https://galgebra.readthedocs.io/)
2. Look at the [examples](../examples/) directory
3. Open an issue on [GitHub](https://github.com/pygae/galgebra/issues)
4. Join the discussion on the pygae community forums

## Contributing

We welcome contributions to improve kingdon compatibility! If you have suggestions or find missing features, please:

1. Open an issue describing the feature
2. Submit a pull request with your implementation
3. Include tests for new functionality

---

*This migration guide was created as part of the effort to improve interoperability between GA libraries in Python.*
