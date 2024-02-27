"""
Represents an uncertain homography matrix `H` with covariance matrix `Σₕ`.

Has 8 degrees of freedom and is full-rank.
Marked `mutable` because can be normalized and hence modified.
"""
mutable struct UncertainHomography{T<:Real}
    H::MMatrix{3,3,T,9}
    Σₕ::SMatrix{9,9,T,81}
end
