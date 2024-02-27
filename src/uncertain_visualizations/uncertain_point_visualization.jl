
"""
    plot_confidence_region(UncertainPoint{Float64}, 0.95)
The last component of `p` is assumed to be constant `1`.

`confidence_level = 1 - α`
"""
function plot_confidence_region(p::UncertainPoint{Float64}, confidence_level::Real)
    χ²_quantile = quantile(_χ²_2_DoF, confidence_level)

    eigenvalues, eigenvectors = eigen((@view p.covariance_matrix[1:2, 1:2]))

    sqrt_d₁ = sqrt(eigenvalues[1])
    sqrt_d₂ = sqrt(eigenvalues[2])

    point_at_angle(θ) = (@view p.point_coords[1:2]) + χ²_quantile * (
        sqrt_d₁ * sin(θ) * eigenvectors[:, 1] + sqrt_d₂ * cos(θ) * eigenvectors[:, 2]
    )

    θ_values = LinRange(0, 2 * pi, 100)
    confidence_region = point_at_angle.(θ_values)

    x_coords, y_coords = map(collect, zip(confidence_region...))
    Plots.plot(x_coords, y_coords)
end
