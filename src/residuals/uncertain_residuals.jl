struct UncertainReprojectionResidual{T<:Real}
    residual::MVector{4,T}
    covariance_matrix::SMatrix{4,4,T,16}
end


struct UncertainForwardResidual{T<:Real}
    residual::MVector{2,T}
    covariance_matrix::SMatrix{2,2,T,4}
end
