export compute_forward_residual, compute_uncertain_forward_residuals


function _compute_forward_residual(v::MVector{2,Float64}, v̂::MVector{2,Float64})::SVector{2,Float64}
    return v - v̂
end


"""
For `u`,`v` both in R² (points of a correspondence in two images) and a Homography `H`, computes the forward residual:

```math
v - project_and_reduce(H * u)
```
"""
function compute_forward_residual(H::MMatrix{3,3,Float64,9}, correspondence::Correspondence{Float64})::SVector{2,Float64}
    v::SVector{2,Float64} = @view correspondence.p₂.point_coords[1:2]
    v̂ = project_and_reduce(apply_homography(H, correspondence.p₁.point_coords))
    return _compute_forward_residual(v, v̂)
end


function compute_uncertain_forward_residuals(UncertainHomography::UncertainHomography{Float64}, correspondences::V)::Vector{UncertainForwardResidual{Float64}} where {V<:AbstractVector{Correspondence{Float64}}}
    @assert length(correspondences) > 0 "No correspondences to compute the residuals for :("
    uncertain_forward_residuals = Vector{UncertainForwardResidual{Float64}}(undef, length(correspondences))

    Σₓ = SmallBlockDiagonal(11, UncertainHomography.Σₕ, (@view correspondences[1].p₁.covariance_matrix[1:2, 1:2]))

    J = zeros(3, 11)
    J[:, 10:11] = @view UncertainHomography.H[:, 1:2]

    for (i, corresp) in enumerate(correspondences)
        p₁ = corresp.p₁
        p₂ = corresp.p₂

        Σₓ[10:11, 10:11] = @view p₁.covariance_matrix[1:2, 1:2]
        
        xₚ = _apply_homography!(UncertainHomography, p₁, Σₓ, J)
        normalize_onto_affine_plane!(xₚ)

        v::MVector{2,Float64} = @view p₂.point_coords[1:2]

        v̂::MVector{2,Float64} = @view xₚ.point_coords[1:2]
        r = _compute_forward_residual(v, v̂)

        Σᵣ::SMatrix{2, 2, Float64, 4} = (@view p₂.covariance_matrix[1:2, 1:2]) + (@view xₚ.covariance_matrix[1:2, 1:2])
        uncertain_forward_residuals[i] = UncertainForwardResidual(r, Σᵣ)
    end
    return uncertain_forward_residuals
end
