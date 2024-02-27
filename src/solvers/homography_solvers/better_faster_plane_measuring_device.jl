"""
Contains the (un)certain homography computation using a novel parametrization proposed by **RH**.
"""

export compute_homography, compute_uncertain_homography


"""
`U` and `V` are matrices of size `3×4`, with correspondences in the columns.

- `U` stands for correspondences in the *first* image.
- `V` stands for correspondences in the *second* image.

Eliminates `1` degree of freedom by enforcing ``λ₄=1``.
"""
function _compute_H!(H_container, U::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T<:Real}
    U₃ = @view U[:, 1:3]
    V₃::SMatrix{3,3,T} = @view V[:, 1:3]

    x_4::SVector{3,T} = U[:, 4]
    x_4ₚ::SVector{3,T} = V[:, 4]

    lu_U₃ = lu(U₃)
    C₃::SVector{3,T} = lu_U₃ \ x_4

    C3ₚ::SVector{3,T} = V₃ \ x_4ₚ

    lambdas::SVector{3,T} = C3ₚ ./ C₃
    Λ₃ = Diagonal(lambdas)

    H_container .= V₃ * Λ₃ * inv(lu_U₃)
    return
end


"""
Takes non-degenerate correspondences. Updates `H_container`. H - homography.

`minimal_set` is flattened with function `correspondences._get_minimal_set_flattened`:

```math
[a₁, b₁, c₁, d₁, a₂, ..., a₄, b₄, c₄, d₄]
```
"""
function _compute_H!(H_container, minimal_set::Arr) where {Arr<:AbstractVector{<:Real}}
    U = similar(minimal_set, 3, 4)
    V = similar(minimal_set, 3, 4)

    U .= 1.0
    V .= 1.0

    for i in 1:4
        U[1:2, i] = @view minimal_set[4*i-3:4*i-2]
        V[1:2, i] = @view minimal_set[4*i-1:4*i]
    end
    _compute_H!(H_container, U, V)
end


"""
Returns the homography `H` only.
"""
function compute_homography(minimal_set::MVector{4,Correspondence{Float64}})::MMatrix{3,3,Float64,9}
    U, V = ones(MMatrix{3,4,Float64}), ones(MMatrix{3,4,Float64})

    for (i, corresp) in enumerate(minimal_set)
        U[1:2, i] = @view corresp.p₁.point_coords[1:2]
        V[1:2, i] = @view corresp.p₂.point_coords[1:2]
    end

    H = zeros(MMatrix{3,3,Float64,9})
    _compute_H!(H, U, V)
    return H
end


"""
Takes the minimal set of correspondences, returns `(H, Jₕ)`
"""
function _compute_H_and_J_H(minimal_set::MVector{4,Correspondence{Float64}})
    all_params_flattened = _get_minimal_set_flattened(minimal_set)
    H = zeros(MMatrix{3,3,Float64,9})
    Jₕ = ForwardDiff.jacobian(_compute_H!, H, all_params_flattened)
    (H, Jₕ)
end


"""
    get_H_and_covariance(MinimalSetCorrespondences)

Takes minimal set and returns an uncertain homography.
"""
function compute_uncertain_homography(minimal_set::MVector{4,Correspondence{Float64}})::UncertainHomography{Float64}
    H, Jₕ = _compute_H_and_J_H(minimal_set)
    cov_X = _get_covariance_matrix_of_minimal_set(minimal_set)
    Σₕ::SMatrix{9,9,Float64,81} = propagate_the_covariance(Jₕ, cov_X)
    UncertainHomography(H, Σₕ)
end
