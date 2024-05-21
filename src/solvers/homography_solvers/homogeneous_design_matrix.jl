"""
Creates a design matrix `A` for Homography (`H`) estimation problems,
where one finds `h:=vec(H)` from the homogeneous system `Ah=0`.
"""


"""
Receives the minimal set as the vector of points using the convention from `_get_minimal_set_flattened`.

`T` can be Float64 or the Dual number used for covariance propagation. 
"""
function get_homogeneous_design_matrix_for_minimal_set(minimal_set::MVector{16,T}) where {T<:Real}
    A = zeros(MMatrix{8,9,T,72})

    for i in 1:4
        a, b = @view minimal_set[4*i-3:4*i-2]
        c, d = @view minimal_set[4*i-1:4*i]

        A[2*i-1, :] = [a, 0, -a * c, b, 0, -b * c, 1, 0, -c]
        A[2*i, :] = [0, a, -a * d, 0, b, -b * d, 0, 1, -d]
    end

    return A
end