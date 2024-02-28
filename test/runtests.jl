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
        residuals = compute_reprojection_residual.(Ref(H), Ref(lu_H), min_set)
        for residual in residuals
            @test isapprox(sum(abs.(residual)), 0.0, atol=1e-6)
        end
    end
end


@testset "Uncertain Residuals" begin
    H = rand(3, 3)
    n_corresps = 1000 + 4
    correspondences = generate_k_random_correspondences(n_corresps, H)
    noised = add_noise(correspondences, 0.1 * I(2), 0.5 * I(2), true)

    min_set::MVector{4,RDR.Correspondence{Float64}} = @view noised[1:4]

    uncertain_H = compute_uncertain_homography(min_set)

    uncertain_reprojection_residuals = compute_uncertain_reprojection_residuals(uncertain_H, (@view noised[5:end]))
    uncertain_forward_residuals = compute_uncertain_forward_residuals(uncertain_H, (@view noised[5:end]))

    @views for (repr_r, forw_r) in zip(uncertain_reprojection_residuals, uncertain_forward_residuals)
        @test repr_r.residual[1:2] ≈ forw_r.residual
        @test repr_r.covariance_matrix[1:2, 1:2] ≈ forw_r.covariance_matrix
    end

    swap_points_in_correspondence!.(noised)
    uncertain_H_inv = compute_uncertain_homography(min_set)
    @test uncertain_H_inv.H ≈ inv(uncertain_H.H)

    uncertain_reprojection_residuals_backward = compute_uncertain_reprojection_residuals(uncertain_H_inv, (@view noised[5:end]))
    uncertain_forward_residuals_backward = compute_uncertain_forward_residuals(uncertain_H_inv, (@view noised[5:end]))

    @views for (repr_r_orig, repr_r_back, forw_r_back) in zip(uncertain_reprojection_residuals, uncertain_reprojection_residuals_backward, uncertain_forward_residuals_backward)
        @test repr_r_orig.residual[1:2] ≈ repr_r_back.residual[3:4]

        @test repr_r_orig.covariance_matrix[1:2, 1:2] ≈ repr_r_back.covariance_matrix[3:4, 3:4]
        @test repr_r_orig.covariance_matrix[3:4, 3:4] ≈ repr_r_back.covariance_matrix[1:2, 1:2]

        @test repr_r_back.residual[1:2] ≈ forw_r_back.residual
        @test forw_r_back.covariance_matrix ≈ repr_r_back.covariance_matrix[1:2, 1:2]
    end
end


@testset "Monte Carlo | Σᵣ verification" begin
    H = 1.0 * I(3)
    runs = 4_000
    noise_matrix = 0.1 * I(2)
    num_corresponodences = 750 + 4

    correspondences = generate_k_random_correspondences(num_corresponodences, H)
    while true
        min_set::MVector{4,RDR.Correspondence{Float64}} = @view correspondences[1:4]
        verify_no_points_are_collinear(min_set, 15.0, 10.0, false) && break
        correspondences = generate_k_random_correspondences(num_corresponodences, H)
    end
    errors = monte_carlo_cov_residuals(correspondences, runs, noise_matrix, false)
    @test median(errors) < 0.05
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

