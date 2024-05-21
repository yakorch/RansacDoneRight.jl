export generate_random_minimal_set, generate_k_random_correspondences, add_noise, swap_points_in_correspondence!

"""
The points are in homogeneous coordinates; covariance is 3×3.
"""
mutable struct UncertainPoint{T<:Real}
    point_coords::MVector{3,T}
    covariance_matrix::MMatrix{3,3,T,9}
end


"""
    Correspondence{T<:Real} is a pair of points along with their covariances.
Represents the same feature from 2 different views.

```math
(x, Σₓ) ↔ (x', Σₓ')
```
"""
mutable struct Correspondence{T<:Real}
    p₁::UncertainPoint{T}
    p₂::UncertainPoint{T}
end


"""
    _generate_random_correspondence(1.0 * I(3), [800.0, 1200.0])
"""
function _generate_random_correspondence(homography, image_bounds)::Correspondence{Float64}
    cov_matrix = zeros(MMatrix{3,3,Float64})
    while true
        u::MVector{2,Float64} = image_bounds .* [rand(), rand()]

        x::MVector{3,Float64} = to_homogeneous(u)
        xₚ::MVector{3,Float64} = project(homography * x)

        if all([0.0, 0.0] .< (xₚ[1:2]) .< image_bounds)
            return Correspondence(
                UncertainPoint(x, cov_matrix),
                UncertainPoint(xₚ, cov_matrix)
            )
        end
    end
end


"""
    generate_k_random_correspondences(k, H, [y, x])

Generates k random correspondences inside an image.

## Examples
```julia-repl
julia> using LinearAlgebra
julia> generate_k_random_correspondences(5, I(3), [1080, 1920])
        <5 correspondences>
```

"""
generate_k_random_correspondences(k::Integer, homography, image_bounds=[800.0, 1200.0])::Vector{Correspondence{Float64}} =
    [_generate_random_correspondence(homography, image_bounds) for _ in 1:k]


"""
Samples 4 uniform points within the image bounds.
"""
function generate_random_minimal_set(image_bounds=[800.0, 1200.0])::MVector{4,Correspondence{Float64}}
    minimal_set = MVector{4,Correspondence{Float64}}(
        Correspondence(
            UncertainPoint(MVector(1.0, 2.0, 3.0), zeros(MMatrix{3,3,Float64})),
            UncertainPoint(MVector(1.0, 2.0, 3.0), zeros(MMatrix{3,3,Float64}))
        ) for _ in 1:4
    )

    generate_random_point() = image_bounds .* rand(2)

    for i in 1:4
        u::MVector{2,Float64}, v::MVector{2,Float64} = (generate_random_point(), generate_random_point())

        minimal_set[i].p₁.point_coords = to_homogeneous(u)
        minimal_set[i].p₂.point_coords = to_homogeneous(v)
    end
    return minimal_set
end


"""
    swap_points_in_correspondence!({x ⟺ x'})

Points in image `1` (`x`) become points in image `2` (`x'`), and vice versa.
"""
function swap_points_in_correspondence!(corresp::Correspondence{Float64})
    corresp.p₁, corresp.p₂ = corresp.p₂, corresp.p₁
    return
end



"""
Let `minimal_set` = ``[Correspondence({[a, b, 1], Σ₁}, {[c, d, 1], Σ₂}), ...]``.

Then the correspondences are flattened as:

```math
[a₁, b₁, c₁, d₁, a₂, ..., a₄, b₄, c₄, d₄]
```
"""
function _get_minimal_set_flattened(minimal_set::MVector{4,Correspondence{Float64}})::MVector{16,Float64}
    all_params_flattened = zeros(MVector{16,Float64}) # Each Correspondence has 4 parameters

    base_index = 0
    for corresp in minimal_set
        all_params_flattened[base_index+1:base_index+2] = @view corresp.p₁.point_coords[1:2]
        all_params_flattened[base_index+3:base_index+4] = @view corresp.p₂.point_coords[1:2]
        base_index += 4
    end

    all_params_flattened
end


"""
Minimal set ⟹ Σₓ, where Σₓ is block-diagonal and of shape 16×16, where each block is 2×2.
 
Has the same order of covariances blocks as in the function `_get_minimal_set_flattened`.
"""
function _get_covariance_matrix_of_minimal_set(minimal_set::V) where {V<:AbstractVector{Correspondence{Float64}}}
    cov_X = zeros(16, 16)

    ind = 1
    for corresp in minimal_set
        cov_X[ind:ind+1, ind:ind+1] = @view corresp.p₁.covariance_matrix[1:2, 1:2]
        cov_X[ind+2:ind+3, ind+2:ind+3] = @view corresp.p₂.covariance_matrix[1:2, 1:2]
        ind += 4
    end

    cov_X
end


"""
    add_noise(corresps, Σ₁, Σ₂, true)
if `actually_add_noise` is `false`, just updates the covariance matrices.

`noise_mᵢ` is 2 × 2 and is the noise covariance matrix of the points at image `i`;
"""
function add_noise(correspondences::V, noise_m_1::M, noise_m_2::M, actually_add_noise::Bool)::V where {V<:AbstractVector{Correspondence{Float64}},M<:AbstractMatrix{Float64}}

    noised_correspondences = similar(correspondences, length(correspondences))

    normal_noise_mean = [0.0, 0.0]

    noise_generator_1 = MvNormal(normal_noise_mean, noise_m_1)
    noise_generator_2 = MvNormal(normal_noise_mean, noise_m_2)

    all_noise_1 = actually_add_noise ? rand(noise_generator_1, length(correspondences)) : zeros(2, length(correspondences))
    all_noise_2 = actually_add_noise ? rand(noise_generator_2, length(correspondences)) : zeros(2, length(correspondences))

    noise_hom_m_1 = zeros(MMatrix{3,3,Float64})
    noise_hom_m_1[1:2, 1:2] .= noise_m_1

    noise_hom_m_2 = zeros(MMatrix{3,3,Float64})
    noise_hom_m_2[1:2, 1:2] .= noise_m_2

    for i in eachindex(correspondences)
        point_1, point_2 = correspondences[i].p₁.point_coords, correspondences[i].p₂.point_coords

        noise_1 = all_noise_1[:, i]
        noise_2 = all_noise_2[:, i]

        noised_p_1 = point_1[1:2] + noise_1
        noised_p_2 = point_2[1:2] + noise_2

        noised_correspondences[i] = Correspondence(
            UncertainPoint(MVector(noised_p_1..., 1), noise_hom_m_1),
            UncertainPoint(MVector(noised_p_2..., 1), noise_hom_m_2)
        )
    end
    noised_correspondences
end


"""
Populates the outliers with random points within the image bounds.
"""
function populate_outliers!(corresps::AbstractVector{Correspondence{Float64}}, image_bounds=[800.0, 1200.0])
    for corresp in corresps
        corresp.p₁.point_coords[1:2] = rand(2) .* image_bounds
        corresp.p₂.point_coords[1:2] = rand(2) .* image_bounds
    end
    return
end
