"""
Modifies `B` with blocks from `matrices` diagonally.
"""
function SmallBlockDiagonal!(B, matrices...)
    start_idx = 1
    for m in matrices
        h, _ = size(m)
        B[start_idx:h+start_idx-1, start_idx:h+start_idx-1] = m
        start_idx += h
    end
end


"""
Does the same as `BlockDiagonal` module but is more efficient for small (`length < 20`) matrices.

Every matrix in `matrices` should be square.
"""
function SmallBlockDiagonal(length, matrices...)
    B = zeros(length, length)
    SmallBlockDiagonal!(B, matrices...)
    B
end


"""
    propagate_the_covariance(J, Σₓ) = Σ_f

For expressions of type `Y = f(X)`, where the covariance of `X` (`Σₓ`) is known.

```math
Σ_f ≈ J × Σₓ × Jᵀ
```
"""
function propagate_the_covariance(J, Σ)
    return J * Σ * J'
end


to_euclidean(x::V) where {V<:AbstractVector{<:Real}} = @view x[1:end-1]

to_homogeneous(x::AbstractVector{T}) where {T<:Real} = vcat(x, one(T))

project(x::AbstractVector{T}) where {T<:Real} = x / x[end]