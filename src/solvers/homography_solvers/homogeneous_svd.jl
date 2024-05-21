using GenericLinearAlgebra


"""
Wrapper for AutoDiff.

Minimal set is flattened with `_get_minimal_set_flattened`.

Uses SVD.
"""
function _populate_homography_svd!(H_container, minimal_set::MVector{16,T}) where {T<:Real}
    A = get_homogeneous_design_matrix_for_minimal_set(minimal_set)
    svd_decomposition = GenericLinearAlgebra.svd(A; full=true)
    h::SVector{9,T} = svd_decomposition.Vt'[:, end]
    H_container .= reshape(h, (3, 3))
end


"""
Uses SVD to recover the null vector of the homogeneous system `Ah = 0`.
"""
function compute_homography_svd(minimal_set::MVector{4,Correspondence{Float64}})::MMatrix{3,3,Float64,9}
    min_set_flattened = _get_minimal_set_flattened(minimal_set)
    H = zeros(MMatrix{3,3,Float64,9})
    _populate_homography_svd!(H, min_set_flattened)
    return H
end


function compute_uncertain_homography_svd(minimal_set::MVector{4,Correspondence{Float64}})::UncertainHomography{Float64}
    min_set_flattened = _get_minimal_set_flattened(minimal_set)
    Σₓ = _get_covariance_matrix_of_minimal_set(minimal_set)

    H = zeros(MMatrix{3,3,Float64,9})
    Jₕ = ForwardDiff.jacobian(_populate_homography_svd!, H, min_set_flattened)

    Σₕ::SMatrix{9,9,Float64,81} = propagate_the_covariance(Jₕ, Σₓ)
    UncertainHomography(H, Σₕ)
end
