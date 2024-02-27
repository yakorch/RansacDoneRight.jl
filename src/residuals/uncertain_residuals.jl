struct UncertainResidual{T<:Real}
    residual::MVector{4,T}
    covariance_matrix::SMatrix{4,4,T,16}
end
