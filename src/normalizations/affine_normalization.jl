"""
For a homogeneous entity `x`, normalizes it onto the affine plane by dividing it by the last component.
Adjusts the covariance matrix accordingly.
"""
function normalize_onto_affine_plane end


"""
Normalizes onto the affine plane and removes the last component.

`π: Rⁿ⁺¹ → Rⁿ`
"""
function π end



function π(v::Union{MVector{3,T},SVector{3,T}})::MVector{2,T} where {T<:Real}
    return v[1:2] / v[3]
end


function π(v::Union{MVector{3,T},SVector{3,T}}, covariance_matrix::Union{MMatrix{3,3,T,9},SMatrix{3,3,T,9}}) where {T<:Real}
    normalized_v = π(v)

    a, b, c = v
    J::SMatrix{2,3,T,6} = (1 / c^2) * [
        c 0 -a;
        0 c -b
    ]

    return normalized_v, propagate_the_covariance(J, covariance_matrix)
end


function π(p::UncertainPoint{Float64})
    return π(p.point_coords, p.covariance_matrix)
end