"""
Parameters of a line as `l=(a, b, c)ᵀ`.

If a homogeneous point `x ∈ P³` lies on the line:

```math
<x, l> = 0
```
"""
mutable struct UncertainLine{T<:Real}
    params::MVector{3,T}
    covariance_matrix::MMatrix{3,3,T,9}
end
