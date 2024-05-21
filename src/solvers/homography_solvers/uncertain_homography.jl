"""
Represents an uncertain homography matrix `H` with covariance matrix `Σₕ`.

Has 8 degrees of freedom and is full-rank.
Marked `mutable` because can be normalized and hence modified.
"""
mutable struct UncertainHomography{T<:Real}
    H::MMatrix{3,3,T,9}
    Σₕ::SMatrix{9,9,T,81}
end


"""
Applies the homography `H` to the point `x`.

Does not perform a projection.

`R³ˣ³ × R³ → R³`
"""
function apply_homography(H::MMatrix{3,3,T,9}, x::MVector{3,T})::MVector{3,T} where {T<:Real}
    return H * x
end


"""
`Σₓ` contains the covariance matrix of the input.

One may get as: `SmallBlockDiagonal(11, UncertainHomography.Σₕ, (@view p.covariance_matrix[1:2, 1:2]))`

`J` is the Jacobian (`3 × 11`) of the transformation. Is modifed in-place.
Last two columns should be the last two columns of `H`.

Used to reduce memory overhead.
"""
function _apply_homography!(UncertainHomography::UncertainHomography{T}, p::UncertainPoint{T}, Σₓ::AbstractMatrix{T}, J::AbstractMatrix{T})::UncertainPoint{T} where {T<:Real}
    mapped_point = apply_homography(UncertainHomography.H, p.point_coords)

    for i in 1:3
        for j in 1:3
            J[j, (i-1)*3+j] = p.point_coords[i]
        end
    end

    Σ::MMatrix{3,3,T,9} = propagate_the_covariance(J, Σₓ)
    return UncertainPoint(mapped_point, Σ)
end
