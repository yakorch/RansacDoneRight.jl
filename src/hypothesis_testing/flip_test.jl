function _compute_det_for_three_points(points::V) where {V<:AbstractVector{UncertainPoint{Float64}}}
    m = vcat([p.point_coords' for p in points]...)
    return det(m)
end


"""
Tests whether the minimal set has been flipped.

Uses the approach proposed by **RH**. `3` correspondences is enough for the test.

If the minimal set isn't flipped, the correspondences are valid.
"""
function is_minimal_set_flipped(minimal_set::MVector{4,Correspondence{Float64}})
    @assert length(minimal_set) == 4 "Wrong number of correspondences"

    three_correspondences = @view minimal_set[1:3]

    first_image_points::SVector{3,UncertainPoint{Float64}} = [corresp.p₁ for corresp in three_correspondences]
    second_image_points::SVector{3,UncertainPoint{Float64}} = [corresp.p₂ for corresp in three_correspondences]

    sign(_compute_det_for_three_points(first_image_points)) != sign(_compute_det_for_three_points(second_image_points))
end
