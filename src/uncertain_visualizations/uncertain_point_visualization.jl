
"""
    plot_confidence_region(UncertainPoint{Float64}, 0.95)
The last component of `p` is assumed to be constant `1`.

`confidence_level = 1 - α`
"""
function plot_confidence_region(p::UncertainPoint{Float64}, confidence_level::Real, existing_plot=nothing)
    χ²_quantile = quantile(_χ²_2_DoF, confidence_level)
    χ_quantile = sqrt(χ²_quantile)

    eigen_decomp = eigen((@view p.covariance_matrix[1:2, 1:2]))
    sqrt_eigenvalues = sqrt.(eigen_decomp.values)

    function point_on_ellipse(θ)
        scaling_vector = χ_quantile .* sqrt_eigenvalues .* [sin(θ), cos(θ)]
        return (@view p.point_coords[1:2]) + eigen_decomp.vectors * scaling_vector
    end

    θ_values = range(0, 2 * pi, length=100)
    ellipse_points = point_on_ellipse.(θ_values)

    y_coords = first.(ellipse_points)
    x_coords = last.(ellipse_points)

    if isnothing(existing_plot)
        return PlotlyJS.plot(x_coords, y_coords)
    else
        new_trace = PlotlyJS.scatter(x=x_coords, y=y_coords, mode="lines")
        add_trace!(existing_plot, new_trace)
    end
end