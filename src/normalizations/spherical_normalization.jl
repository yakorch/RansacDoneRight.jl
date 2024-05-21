

"""
Scales the uncertain (random) entity onto the unit sphere.
Usually used for homogeneous entities.

If `length(elements) = N`, the `covariance_matrix` must be `N × N`.

Returns normalized elements and the approximation of the covariance matrix.
"""
function normalize_onto_unit_sphere
end

export normalize_onto_unit_sphere, normalize_onto_unit_sphere!


function normalize_onto_unit_sphere(v::V, covariance_matrix::M) where {V<:AbstractVector{<:Real},M<:AbstractMatrix{<:Real}}
    one_over_norm = 1 / sqrt(sum(v .^ 2))

    normalized_v = v .* one_over_norm

    J = one_over_norm * (I(length(v)) - normalized_v * normalized_v')  # J = (1/|v|)(I - vv^T/|v|). Easy to derive.

    normalized_v, propagate_the_covariance(J, covariance_matrix)
end


function normalize_onto_unit_sphere(matrix::M) where {M<:AbstractMatrix{<:Real}}
    return matrix / LinearAlgebra.norm(matrix)
end


function normalize_onto_unit_sphere(matrix::M₁, covariance_matrix::M₂) where {M₁<:AbstractMatrix{<:Real},M₂<:AbstractMatrix{<:Real}}
    v = vec(matrix)
    normalized_v, Σₛ = normalize_onto_unit_sphere(v, covariance_matrix)
    return reshape(normalized_v, size(matrix)), Σₛ
end


function normalize_onto_unit_sphere!(uncertain_homography::UncertainHomography{Float64})
    uncertain_homography.H, uncertain_homography.Σₕ = normalize_onto_unit_sphere(uncertain_homography.H, uncertain_homography.Σₕ)
    return
end


function normalize_onto_unit_sphere(uncertain_homography::UncertainHomography{Float64})
    normalized_homography = UncertainHomography(uncertain_homography.H, uncertain_homography.Σₕ)
    normalize_onto_unit_sphere!(normalized_homography)
    return normalized_homography
end


function normalize_onto_unit_sphere!(l::UncertainLine{Float64})
    l.params, l.covariance_matrix = normalize_onto_unit_sphere(l.params, l.covariance_matrix)
    return
end


function normalize_onto_unit_sphere!(p::UncertainPoint{Float64})
    p.point_coords, p.covariance_matrix = normalize_onto_unit_sphere(p.point_coords, p.covariance_matrix)
    return
end


function normalize_onto_unit_sphere(p::UncertainPoint{Float64})
    normalized_p = UncertainPoint(p.point_coords, p.covariance_matrix)
    normalize_onto_unit_sphere!(normalized_p)
    normalized_p
end
