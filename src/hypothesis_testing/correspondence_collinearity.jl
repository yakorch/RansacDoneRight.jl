"""
Verifies the degeneracy of a minimal set of correspondences for Homography `H` computation.
"""


const _χ²_1_DoF = Chisq(1)
const _χ²_2_DoF = Chisq(2)
const _χ²_4_DoF = Chisq(4)


export verify_no_points_are_collinear

"""
Computes the statistic of `2` image points identity.

Follows a Chi-squared distribution with `2` deg. of freedom (only considers `x` and `y` coordinates.)

Two points are identical (`p₁ ≡ p₂`) when the statistic is close to `0`.
"""
function compute_identity_statistic(p₁::UncertainPoint{Float64}, p₂::UncertainPoint{Float64})::Float64
    μ₁::SVector{2,Float64} = @view p₁.point_coords[1:2]
    μ₂::SVector{2,Float64} = @view p₂.point_coords[1:2]
    coords_difference::SVector{2,Float64} = μ₁ - μ₂  # E(d) = μ₁ - μ₂

    Σ₁::SMatrix{2,2,Float64} = @view p₁.covariance_matrix[1:2, 1:2]
    Σ₂::SMatrix{2,2,Float64} = @view p₁.covariance_matrix[1:2, 1:2]
    difference_covariance_matrix::SMatrix{2,2,Float64} = Σ₁ + Σ₂  # Var(d) = Σ₁ + Σ₂

    (coords_difference'*inv(difference_covariance_matrix)*coords_difference)[1, 1]  # Mahalanobis distance
end


"""
For an expression of type

```math
c = x × y
```
returns the skew-symmetric matrix `Sₓ` s.t. 

```math
c = Sₓy
```
"""
function get_skew_symmetric_matrix_for_cross_product(x::MVector{3,Float64})::SMatrix{3,3,Float64}
    a, b, c = x
    [
        0 -c b;
        c 0 -a;
        -b a 0
    ]
end


"""
Assumes two points are independent.

Uses the conventions introduced by Forstner:
https://www.ipb.uni-bonn.de/pdfs/Forstner2004Uncertainty.pdf
page 23

Does not perform spherical normalization.
"""
function find_uncertain_line(p₁::UncertainPoint{Float64}, p₂::UncertainPoint{Float64})::UncertainLine{Float64}
    Sₓ = get_skew_symmetric_matrix_for_cross_product(p₁.point_coords)
    y = p₂.point_coords

    l::MVector{3,Float64} = Sₓ * y

    Uₐ = Sₓ
    Vᵦ = get_skew_symmetric_matrix_for_cross_product(p₂.point_coords)

    Σₗ::MMatrix{3,3,Float64,9} = Vᵦ * p₁.covariance_matrix * Vᵦ' + Uₐ * p₂.covariance_matrix * Uₐ'   # expension of cov. matrix after block multiplicaiton.

    UncertainLine(l, Σₗ)
end



"""
Computes the statistic of incidence of the image point and the line.
Assumes the independence of the point and the line.

Follows a Chi-squared distribution with `1` deg. of freedom.

`p ∈ l` ~ the statistic is close to `0`.


Uses the conventions introduced by Forstner:
https://www.ipb.uni-bonn.de/pdfs/Forstner2004Uncertainty.pdf
"""
function compute_incidence_statistic(p::UncertainPoint{Float64}, l::UncertainLine{Float64})::Float64
    c = (p.point_coords'*l.params)[1, 1]   # <x, l>

    J₁ = l.params'
    J₂ = p.point_coords'

    var_c::Float64 = (J₁*p.covariance_matrix*J₁'+J₂*l.covariance_matrix*J₂')[1, 1]

    return c^2 / var_c
end



"""
T₁ and T₂ are the same as in function `verify_no_points_are_collinear`.
"""
function are_three_points_collinear(p₁::UncertainPoint{Float64}, p₂::UncertainPoint{Float64}, p₃::UncertainPoint{Float64}, T₁::Float64, T₂::Float64)
    compute_identity_statistic(p₂, p₃) < T₁ && return true
    l = find_uncertain_line(p₂, p₃)
    return compute_incidence_statistic(p₁, l) < T₂
end


"""
Returns `true` if no points in the minimal set are collinear, and `false` otherwise.

- `T₁` is the threshold statistic used for 2 points indentity test. Is some `quantile(χ²₂)`. For example `T₁=5.99` corresponds to confidence level of 95%.

- `T₂` is the threshold statistic used for point-line incidence test. Is some `quantile(χ²₁)`. For example `T₂=3.84` corresponds to confidence level of 95%.

- `check_flip` tells whether to check for the flip between the correspondences.
"""
function verify_no_points_are_collinear(minimal_set::MVector{4,Correspondence{Float64}}, T₁::Float64, T₂::Float64, check_flip::Bool)::Bool

    check_flip && is_minimal_set_flipped(minimal_set) && return false

    for triplet in combinations(minimal_set, 3)
        for i in (:p₁, :p₂)
            are_three_points_collinear(getfield(triplet[1], i), getfield(triplet[2], i), getfield(triplet[3], i), T₁, T₂) && return false
        end
    end

    return true
end
