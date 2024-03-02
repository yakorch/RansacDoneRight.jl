"""
For a homogeneous entity `x`, normalizes it onto the affine plane by dividing it by the last component.
Adjusts the covariance matrix accordingly.
"""
function normalize_onto_affine_plane end


function normalize_onto_affine_plane!(p::UncertainPoint{Float64})
    a, b, c = p.point_coords

    J::SMatrix{2,3,Float64,6} = (1 / c^2) * [
        c 0 -a;
        0 c -b
    ]

    p.covariance_matrix[1:2, 1:2] = propagate_the_covariance(J, p.covariance_matrix)

    p.covariance_matrix[3, :] .= 0.0
    p.covariance_matrix[:, 3] .= 0.0

    p.point_coords /= p.point_coords[end]
    return
end


"""
Normalizes onto the affine plane and removes the last component (reduces dimensionality).

`π: Rⁿ⁺¹ → Rⁿ`
"""
function project_and_reduce end


function project_and_reduce(v::Union{MVector{3,T},SVector{3,T}})::MVector{2,T} where {T<:Real}
    return v[1:2] / v[3]
end


function project_and_reduce(v::Union{MVector{3,T},SVector{3,T}}, covariance_matrix::Union{MMatrix{3,3,T,9},SMatrix{3,3,T,9}}) where {T<:Real}
    normalized_v = project_and_reduce(v)

    a, b, c = v
    J::SMatrix{2,3,T,6} = (1 / c^2) * [
        c 0 -a;
        0 c -b
    ]

    return normalized_v, propagate_the_covariance(J, covariance_matrix)
end


function project_and_reduce(p::UncertainPoint{Float64})
    return project_and_reduce(p.point_coords, p.covariance_matrix)
end