"""
Residuals along with covariance matrices.
Assumed to follow a Multivariate Gaussian distribution.
"""
abstract type UncertainResidual{T<:Real} end


struct UncertainReprojectionResidual{T<:Real} <: UncertainResidual{T}
    residual::MVector{4,T}
    covariance_matrix::SMatrix{4,4,T,16}
end


struct UncertainForwardResidual{T<:Real} <: UncertainResidual{T}
    residual::SVector{2,T}
    covariance_matrix::SMatrix{2,2,T,4}
end
