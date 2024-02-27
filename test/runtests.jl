using RansacDoneRight
using Test
import RansacDoneRight as RDR

using LinearAlgebra
using StaticArrays
using Statistics


@testset "RansacDoneRight.jl" begin

end


@testset "Homography Solver" begin
    _TEST_REPETITIONS = 100

    for _ in 1:_TEST_REPETITIONS
        min_set = generate_random_minimal_set()
        H = compute_homography(min_set)
        uncertain_H = compute_uncertain_homography(min_set)

        @test H ≈ uncertain_H.H

        lu_H = lu(H)
        residuals = compute_residual.(Ref(H), Ref(lu_H), min_set)
        for residual in residuals
            @test isapprox(sum(abs.(residual)), 0.0, atol=1e-6)
        end
    end
end

@testset "Monte Carlo | Σᵣ verification" begin
    H = 1.0 * I(3)
    runs = 5_000
    noise_matrix = 0.1 * I(2)
    num_corresponodences = 1_000 + 4

    correspondences = generate_k_random_correspondences(num_corresponodences, H)
    while true
        min_set::MVector{4,RDR.Correspondence{Float64}} = @view correspondences[1:4]
        verify_no_points_are_collinear(min_set, 15.0, 10.0, false) && break
        correspondences = generate_k_random_correspondences(num_corresponodences, H)
    end
    errors = monte_carlo_cov_residuals(correspondences, runs, noise_matrix, false)
    @test mean(errors) < 0.05
end


@testset "Spherical Normalization" begin
    a = [4]
    cov_a = I(1)
    normed_a, normed_cov_a = normalize_onto_unit_sphere(a, cov_a)

    @test normed_a ≈ [1.0]
    @test normed_cov_a[1, 1] ≈ 0.0

    v = [6, -8]
    cov_v = I(2)

    normed_v, normed_cov_v = normalize_onto_unit_sphere(v, cov_v)
    @test normed_v ≈ [0.6, -0.8]
    @test isapprox(sum(abs.(normed_cov_v * normed_v)), 0.0, atol=1e-10)
end
