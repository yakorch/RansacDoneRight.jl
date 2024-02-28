
export compute_inlier_mask


"""
    compute_inlier_test_statistic(UncertainResidual{Float64})
Uncertain residual is assumed to have a mean at `0` and follow a Multivariate Gaussian distribution.

Calculates the Mahalanobis distance.

One may use `Distributions.quantile(_χ²_k_DoF, confidence_level)` to get the threshold for the test.
"""
function compute_inlier_test_statistic(uncertain_residual::UncertainResidual{Float64})::Float64
    r = uncertain_residual.residual

    # TODO: benchmark inverse vs. LU decomposition
    mahalanobis_dist = r' * (lu(uncertain_residual.covariance_matrix) \ r)
    @assert isfinite(mahalanobis_dist) "Mahalanobis distance is not finite. Got: $mahalanobis_dist. Residual: $r. Covariance: $(uncertain_residual.covariance_matrix)."

    return mahalanobis_dist
end


"""
Performs the following test:

`H₀`: Correspondence is an inlier
`H₁`: Correspondence is an outlier.

mask contains `true` where H₀ wasn't rejected.

For **Reprojection**/**Forward** residual, `T` is the test statistic corresponding to the quantile of the `χ`-squared distribution with **`4`**/**`2`** deg. of freedom. **`9.49`**/**`5.99`** corresponds to a confidence level of `0.95`.
"""
function compute_inlier_mask(uncertain_residuals::V, T::Float64) where {V<:AbstractVector{UncertainResidual{Float64}}}
    @. compute_inlier_test_statistic(uncertain_residuals) < T
end
