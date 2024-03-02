export compute_reprojection_residual, compute_uncertain_reprojection_residuals


"""
Saves the residual in `r_container` ∈ R⁴.
"""
function _compute_reprojection_residual!(r_container::MVector{4,T}, H, lu_H::Union{LinearAlgebra.LU,StaticArrays.LU}, u::SVector{2,T}, v::SVector{2,T}) where {T<:Real}
    # H_times_u::SVector{3,T} = H * to_homogeneous(u)
    H_times_u::SVector{3,T} = [
        H[1, 1] * u[1] + H[1, 2] * u[2] + H[1, 3],
        H[2, 1] * u[1] + H[2, 2] * u[2] + H[2, 3],
        H[3, 1] * u[1] + H[3, 2] * u[2] + H[3, 3]
    ]

    # Finds `y` in `H⁻¹ * [v, 1]^T = y`. Because is equivalent to `H * y = [v, 1]^T`.
    H_inv_times_v::SVector{3,T} = lu_H \ to_homogeneous(v)

    r_container[1:2] = v - to_euclidean(project(H_times_u))
    r_container[3:4] = u - to_euclidean(project(H_inv_times_v))
    return
end


"""
Residual - 2 stacked distance vectors. Includes the symmetric error.

`u` and `v` ∈ R² -- points on the first and the second image, respectively.
"""
function _compute_reprojection_residual(H, lu_H::Union{LinearAlgebra.LU,StaticArrays.LU}, u::SVector{2,T}, v::SVector{2,T}) where {T<:Real}
    r = zeros(MVector{4,Float64})
    _compute_reprojection_residual!(r, H, lu_H, u, v)
    return r
end


function compute_reprojection_residual(H::MMatrix{3,3,Float64,9}, lu_H::StaticArrays.LU, correspondence::Correspondence{Float64})
    u::SVector{2,Float64} = @view correspondence.p₁.point_coords[1:2]
    v::SVector{2,Float64} = @view correspondence.p₂.point_coords[1:2]
    return _compute_reprojection_residual(H, lu_H, u, v)
end


"""
A wrapper for `AutoDiff` pkg to work.

`h_and_points`=``[vec(H), a, b, c, d]``
"""
function _compute_reprojection_residual_autodiff_wrapper!(r_container::MVector{4,T}, h_and_points::MVector{13,T}) where {T<:Real}
    h::SVector{9,T} = @view h_and_points[1:9]
    u::SVector{2,T}, v::SVector{2,T} = (@view h_and_points[10:11]), (@view h_and_points[12:end])

    H = reshape(h, (3, 3))

    _compute_reprojection_residual!(r_container, H, lu(H), u, v)
    return
end


"""
Returns the residuals along with respective covariances.


`correspondences` and uncertain `H` are assumed to be independent.
"""
function compute_uncertain_reprojection_residuals(uncertain_H::UncertainHomography{Float64}, correspondences::V)::Vector{UncertainReprojectionResidual{Float64}} where {V<:AbstractVector{Correspondence{Float64}}}
    @assert length(correspondences) > 0 "No correspondences to compute the residuals for :("

    uncertain_residuals = Vector{UncertainReprojectionResidual{Float64}}(undef, length(correspondences))

    arguments_vector::MVector{13,Float64} = [vec(uncertain_H.H)..., 0.0, 0.0, 0.0, 0.0]

    Σ = SmallBlockDiagonal(13, uncertain_H.Σₕ)

    for (i, corresp) in enumerate(correspondences)
        arguments_vector[10:11] = @view corresp.p₁.point_coords[1:2]
        arguments_vector[12:13] = @view corresp.p₂.point_coords[1:2]

        residual = zeros(MVector{4,Float64})

        Jᵣ = ForwardDiff.jacobian(_compute_reprojection_residual_autodiff_wrapper!, residual, arguments_vector)

        euc_cov_1 = @view corresp.p₁.covariance_matrix[1:2, 1:2]
        euc_cov_2 = @view corresp.p₂.covariance_matrix[1:2, 1:2]

        Σ[10:11, 10:11] = euc_cov_1
        Σ[12:13, 12:13] = euc_cov_2

        Σᵣ::SMatrix{4,4,Float64,16} = propagate_the_covariance(Jᵣ, Σ)
        uncertain_residuals[i] = UncertainReprojectionResidual(residual, Σᵣ)
    end
    uncertain_residuals
end
