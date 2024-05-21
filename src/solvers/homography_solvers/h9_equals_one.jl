

"""
Wrapper for AutoDiff.

Minimal set is flattened with `_get_minimal_set_flattened`.
Uses LU decomposition.
"""
function _populate_homography_h₉_equals_1!(H_container, minimal_set::MVector{16,T}) where {T<:Real}
    A = get_homogeneous_design_matrix_for_minimal_set(minimal_set)
    B = @view A[:, 1:8]
    a::SVector{8,T} = A[:, 9]

    h::SVector{8,T} = -(lu(B) \ a)

    H_container .= reshape(push(h, one(T)), 3, 3)
    return
end


"""
Homography solver that intoruduces degeneracy by setting the parameter `h₉=1`.
"""
function compute_homography_h₉_equals_1(minimal_set::MVector{4,Correspondence{Float64}})::MMatrix{3,3,Float64,9}
    min_set_flattened = _get_minimal_set_flattened(minimal_set)
    H = zeros(MMatrix{3,3,Float64,9})
    _populate_homography_h₉_equals_1!(H, min_set_flattened)
    return H
end


function compute_uncertain_homography_h₉_equals_1(minimal_set::MVector{4,Correspondence{Float64}})::UncertainHomography{Float64}
    min_set_flattened = _get_minimal_set_flattened(minimal_set)
    Σₓ = _get_covariance_matrix_of_minimal_set(minimal_set)

    H = zeros(MMatrix{3,3,Float64,9})
    Jₕ = ForwardDiff.jacobian(_populate_homography_h₉_equals_1!, H, min_set_flattened)
    Σₕ::SMatrix{9,9,Float64,81} = propagate_the_covariance(Jₕ, Σₓ)
    UncertainHomography(H, Σₕ)
end


using LinearAlgebra

A = [
    1 0 0.8 0.2;
    0 2 0.4 0;
    0.8 0.4 1 -0.5;
    0.2 0 -0.5 3
]

rank(
    A
)