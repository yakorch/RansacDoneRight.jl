export monte_carlo_cov_residuals


"""
Runs the Monte Carlo to estimate the accuracy of First-Order Error Propagation for the covariance of the residuals.

# Arguments:
    - `truth_correspondences` -- unnoised correspondences
    - `run_each_corresp` -- number of runs
    - `noise_matrix` -- the 2 × 2 covariance matrix of the points' noise.
    - `verbose` -- if `true`, displays large residuals when encountered and adds extra prints for Debug.

Parallelized version (uses `Threads`).
"""
function monte_carlo_cov_residuals(truth_correspondences::V, run_each_corresp::Integer, noise_matrix, verbose=true) where {V<:AbstractVector{Correspondence{Float64}}}

    truth_minset_unrealized_noise = add_noise(truth_correspondences, noise_matrix, noise_matrix, false)
    truth_min_set::MVector{4,Correspondence{Float64}} = @view truth_minset_unrealized_noise[1:4]

    uncertain_H = compute_uncertain_homography(truth_min_set)
    uncertain_residuals = compute_uncertain_residuals(uncertain_H, @view truth_minset_unrealized_noise[5:end])
    propagated_covs = [el.covariance_matrix for el in uncertain_residuals]

    # -4 because we don't compute the residuals on the minimal set
    num_residuals = length(truth_correspondences) - 4
    @assert num_residuals > 0 "Not enough correspondences"

    num_threads = Threads.nthreads()
    thread_covariances = [Matrix{Float64}[zeros(4, 4) for _ in 1:num_residuals] for _ in 1:num_threads]

    Threads.@threads for _ in 1:run_each_corresp
        noised_correspondences = add_noise(truth_correspondences, noise_matrix, noise_matrix, true)
        min_set::MVector{4,Correspondence{Float64}} = @view noised_correspondences[1:4]
        H = compute_homography(min_set)
        lu_H = lu(H)

        covariances::Vector{Matrix{Float64}} = thread_covariances[Threads.threadid()]

        for j in 1:num_residuals
            residual = compute_residual(H, lu_H, noised_correspondences[j+4])

            covariances[j] += residual * residual'

            if (verbose && sum(abs.(residual)) > 2e6)
                display(H)
                println("THREAD $(Threads.threadid()) found a too large residual")
            end
        end
    end

    errors = Array{Float64}(undef, num_residuals)
    for i in 1:num_residuals
        monte_cov_m = sum((thread_covariances[t][i] for t in 1:num_threads)) ./ run_each_corresp
        errors[i] = norm(monte_cov_m - propagated_covs[i]) / norm(monte_cov_m)

        if verbose
            println("===")
            println("Monte Carlo: ")
            display(monte_cov_m)
            println("Propagation: ")
            display(propagated_covs[i])
            println("Relative error: ", round(100 * errors[i]; digits=3), "%")
            println("===")
        end
    end
    errors
end



"""
Runs Monte Carlo a lot of times and returns a DataFrame with results
"""
function run_monte_carlo_comparison()::DataFrame
    monte_carlo_comparison_df = DataFrame(Sigma_1=[], Sigma_2=[], Corr_Coeff=[], Rel_Error=[])

    sigmas_1 = [0.5, 1, 2.0, 3.0]
    sigmas_2 = [0.5, 1, 2.0, 3.0]
    rhos = [0, 0.25, 0.75]

    total_runs_per_one_variant::Integer = 1e4
    sets_correspondences::Integer = 1
    one_set_runs::Integer = total_runs_per_one_variant / sets_correspondences

    all_variations = collect(Base.Iterators.product(sigmas_1, sigmas_2, rhos))

    for (s_1, s_2, rho) in all_variations
        noise_m = [
            s_1^2 (rho*s_1*s_2);
            (rho*s_1*s_2) s_2^2
        ]

        for _ in 1:sets_correspondences
            correspondences = generate_k_random_correspondences(25 + 4, [1.1 0.1 15.0; -0.3 0.9 -10.0; 0.001 -0.001 1.1])
            rel_errors = monte_carlo_cov_residuals(correspondences, one_set_runs, noise_m, false)
            for error in rel_errors
                push!(monte_carlo_comparison_df, (s_1, s_2, rho, error))
            end
        end
    end
    monte_carlo_comparison_df
end
