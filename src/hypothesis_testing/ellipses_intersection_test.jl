"""
Cov-RANSAC approach uses ellipses intersection for consensus set labelling.

If two ellipses (confidence regions of Multivariate Normal dist.) of the transferred point and the ground-truth point intersect, the correspondence is considered an inlier.
"""

using NLsolve


"""
Returns symbolic function for the point's `χ²₂` statistic.

Implements the labelling function for the Cov-RANSAC approach:
    https://www.researchgate.net/publication/221110087_Exploiting_Uncertainty_in_Random_Sample_Consensus

`p` is the ground-truth point, and `p̂` is the transferred point.
"""
function compute_CovRANSAC_labelling_statistic(p::UncertainPoint{Float64}, p̂::UncertainPoint{Float64})
    u::SVector{2,Float64} = @view p.point_coords[1:2]
    û::SVector{2,Float64} = @view p̂.point_coords[1:2]

    Σ⁻¹_p = inv((@view p.covariance_matrix[1:2, 1:2]))
    Σ⁻¹_p̂ = inv((@view p̂.covariance_matrix[1:2, 1:2]))

    function system!(F, x)
        _a, _b, _λ = x

        r₁::SVector{2,Float64} = [_a - u[1], _b - u[2]]
        r₂::SVector{2,Float64} = [_a - û[1], _b - û[2]]

        Σ⁻¹_times_r₁::SVector{2,Float64} = Σ⁻¹_p * r₁
        Σ⁻¹_times_r₂::SVector{2,Float64} = Σ⁻¹_p̂ * r₂

        D₁::Float64 = r₁' * Σ⁻¹_times_r₁
        D₂::Float64 = r₂' * Σ⁻¹_times_r₂

        ∇D₁::SVector{2,Float64} = 2 * Σ⁻¹_times_r₁
        ∇D₂::SVector{2,Float64} = 2 * Σ⁻¹_times_r₂

        F[1] = D₁ - D₂
        F[2] = ∇D₁[1] - _λ * ∇D₂[1]
        F[3] = ∇D₁[2] - _λ * ∇D₂[2]
    end

    while true
        coeff = rand()
        x0::MVector{3,Float64} = [u[1] * coeff + û[1] * (1 - coeff), u[2] * coeff + û[2] * (1 - coeff), -6.5 * rand()]
        sol = nlsolve(system!, x0; iterations=400)

        λ = sol.zero[3]

        λ > 0 && continue

        x::SVector{2,Float64} = sol.zero[1:2]
        D = (x - u)' * Σ⁻¹_p * (x - u)
        return sol.zero, D
    end
end



function compute_CovRANSAC_labelling_statistics(UH::UncertainHomography{Float64}, correspondences::V) where {V<:AbstractVector{Correspondence{Float64}}}
    statistics = Vector{Float64}(undef, length(correspondences))

    J = zeros(3, 11)
    J[:, 10:11] = UH.H[:, 1:2]

    Σ = SmallBlockDiagonal(11, UH.Σₕ)

    for (i, corresp) in enumerate(correspondences)
        Σ[10:11, 10:11] = corresp.p₁.covariance_matrix[1:2, 1:2]
        xₚ = _apply_homography!(UH, corresp.p₁, Σ, J)
        normalize_onto_affine_plane!(xₚ)

        _, T = compute_CovRANSAC_labelling_statistic(corresp.p₂, xₚ)
        statistics[i] = T
    end
    statistics
end
