"""
Plots 2-view 2D scene.

`image_bounds` assumed to be (y, x).

`errors` is an error of the point: could represent a relative error from the Monte Carlo. Should be in the same order as `correspondences` vector.

If no `errors` provided, doesn't plot the errors.

"""
function plot_correspondences(correspondences::V; errors=[], image_bounds=[800.0, 1200.0]) where {V<:AbstractVector{Correspondence{Float64}}}
    p1_plot = Plots.plot(legend=false, xlim=(-10, image_bounds[2] + 10), ylim=(-10, image_bounds[1] + 10), title="Image 1", size=(1000, 400))
    p2_plot = Plots.plot(legend=false, xlim=(-10, image_bounds[2] + 10), ylim=(-10, image_bounds[1] + 10), title="Image 2", size=(1000, 400))

    sorted_indices = sortperm(correspondences, by=(corresp -> corresp.p₁.point_coords[1]))
    num_points = length(correspondences)

    if length(errors) == 0
        color_gradient = [RGB(1 - i / num_points, 0.7, 0.9) for i in 1:num_points]
    else
        max_err = maximum(errors)
        min_err = minimum(errors)
        error_percentiles = [(err - min_err) / (max_err - min_err) for err in errors]

        color_gradient = [RGB(err_percentile, 0.5 - err_percentile / 2, 0.55) for err_percentile in error_percentiles]
    end

    for i in 1:num_points
        sorted_ind = sorted_indices[i]

        color = color_gradient[i]
        if length(errors) == 0
            corresp = correspondences[sorted_ind]
        else
            corresp = correspondences[i]
        end

        p1 = corresp.p₁.point_coords
        p2 = corresp.p₂.point_coords

        # scatter the points; (x, y) order
        Plots.scatter!(p1_plot, [p1[2]], [p1[1]], color=color, label=false)
        Plots.scatter!(p2_plot, [p2[2]], [p2[1]], color=color, label=false)

    end
    Plots.plot(p1_plot, p2_plot, layout=(1, 2))
end