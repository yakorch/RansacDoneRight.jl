### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ c4f95e90-a3f7-11ee-179e-119f144e6845
begin
    using Pkg
	NOTEBOOKS_PATH = "src/pluto_notebooks"
	DATA_PATH = "data"  # store generated by notebook data
    @assert endswith(pwd(), NOTEBOOKS_PATH) "Wrong directory!"
	RDR_PATH = "../.."  # path to the RDR project root
    Pkg.activate(RDR_PATH)
	Pkg.instantiate()
	
	import RansacDoneRight as RDR
    md"""
    ### Activating RansacDoneRight environment...
    """
end

# ╔═╡ 96ee9af6-0fd5-46f7-a68b-a00a1bd47b6c
begin
    using StaticArrays
    using DataFrames
    using LinearAlgebra
    using Revise
    using Plots
    using PlutoUI
    using CSV
    using Distributions
end

# ╔═╡ ce1da846-d7f2-4cc2-bae7-f3446dbf7dec
md"""
#### Custom Pluto Settings -->
"""

# ╔═╡ 8bbd7875-665f-4949-87d0-a8a2d1ebdfb9
begin
    html"""
    <style>
    	main {
    		margin: 0 auto;
    		max-width: 2000px;
        	padding-left: max(160px, 15%);
        	padding-right: max(160px, 15%);
    	}
    </style>
    """
end


# ╔═╡ 6286baf3-1124-4992-8779-c84e4a8b82dc
begin
    n_corresps_slider = @bind n_corresps Slider(5:1:80, 25, true)
    md"""
    # Number of Correspondences:

    N = $(n_corresps_slider)
    """
end

# ╔═╡ 1632b500-d0d9-47ca-be39-ad6b60d4817c
ground_truth_H = [
    1.15 0.1 5;
    0.25 0.9 -10;
    0.0003 0.0005 0.99
]

# ╔═╡ 9863370c-c52c-4906-9b75-720b475d38f3
all_correspondences = RDR.generate_k_random_correspondences(n_corresps, ground_truth_H)

# ╔═╡ 23e47540-5958-4ef3-9ae5-bc0b1222318d
begin
    sigma_1_slider = @bind sigma_1 Slider(0.2:0.2:10, 3.2, true)
    silder_sigma_2 = @bind sigma_2 Slider(0.2:0.2:10, 2.3, true)
    rho_slider = @bind rho Slider(0:0.02:0.99, 0.01, true)


    md"""
    # Noise Covariance Matrix:

    σ₁ = $(sigma_1_slider) 

    σ₂= $(silder_sigma_2)

    ρ = $(rho_slider)
    """
end

# ╔═╡ abd25bf2-ba31-4d9f-a674-b542e387824e
noise_cov_matrix = [
    sigma_1^2 (sigma_1*sigma_2*rho);
    (sigma_1*sigma_2*rho) sigma_2^2
]

# ╔═╡ c00895e5-480f-4aba-9474-4d4f6229be51
begin
    noised_correspondences = RDR.add_noise(all_correspondences, noise_cov_matrix, noise_cov_matrix, true)
    
    minimal_set_truth::MVector{4, RDR.Correspondence{Float64}} = sample(all_correspondences, 4; replace=false)
    minimal_set_noised::MVector{4, RDR.Correspondence{Float64}} = sample(noised_correspondences, 4; replace=false)
    md"""
    ## Adding Noise...
    """
end

# ╔═╡ ac952f41-be38-42a8-b2dc-95a244a6c5aa
RDR.plot_correspondences(noised_correspondences)

# ╔═╡ 63de67ba-2f64-43c6-91f4-4f6164ebf5f9
begin
    uncertain_homography = RDR.compute_uncertain_homography(minimal_set_noised)
    uncertain_homography.H, uncertain_homography.Σₕ
end

# ╔═╡ 63b1567c-85cb-4def-a53d-48ba1ac15ced
RDR.compute_uncertain_residuals(uncertain_homography, noised_correspondences)

# ╔═╡ 57adeb03-59ed-4e28-ab47-46c7209f9fd8
md"""
# Monte Carlo Simulations
"""

# ╔═╡ 5133d461-947f-4451-8478-a1cb0b9aef7d
begin
    one_long_run_times = 50_000
    md"""
       #### Running Monte Carlo $(one_long_run_times) times. 
       """
end

# ╔═╡ de7d650f-10a3-4b06-bd56-8e815d443d98
begin
    curr_cov_matrix = [
        1 0.5;
        0.5 1]
    long_run = RDR.monte_carlo_cov_residuals(all_correspondences, one_long_run_times, curr_cov_matrix, false)

    Plots.plot(100 .* long_run, seriestype=:histogram, bins=length(all_correspondences) - 4, size=(800, 600), xlabel="Relative Error, %", ylabel="Frequency", label="Monte Carlo $(one_long_run_times) times vs First-Order Error Prop.")
end

# ╔═╡ 80e65604-06ac-48bf-a792-8e042b471466
begin
    md"""
    #### Error illustrated on points:
    $(RDR.plot_correspondences(all_correspondences[5:end]; errors=long_run))
    More red indicates a larger error
    """
end

# ╔═╡ 0633345c-cf84-4b57-8d28-d7a513b0285a
md"""
#### Minimal set used:
$(RDR.plot_correspondences(all_correspondences[1:4]))"""

# ╔═╡ 5c317ad2-7035-429e-ad67-8592d87c201c
begin
    bind_check_run_sims = @bind run_simulations Select([true, false], default=false)
    md"""
    ### Want to rerun the simulations?: $(bind_check_run_sims)
    """
end


# ╔═╡ 99a71d7f-5be8-4aa0-99f5-b588d57dae0e
begin
    simulation_path = DATA_PATH * "/" * "error_prop_error_vs_monte_carlo.csv"
    if run_simulations
        df_monte = RDR.run_monte_carlo_comparison()
        CSV.write(simulation_path, df_monte)
    end

    df_monte = DataFrame(CSV.File(simulation_path))
    sigma_1_values = unique(df_monte.Sigma_1)
    sigma_2_values = unique(df_monte.Sigma_2)
    corr_coeff_values = unique(df_monte.Corr_Coeff)

    md"""
    Deciding whether to run the simulations... **$(run_simulations)**
    """
end

# ╔═╡ ead52d9e-3dad-4a15-919f-727f695f3529
begin
    sigma_1_chooser = @bind sigma_1_chosen Select(sigma_1_values)
    sigma_2_chooser = @bind sigma_2_chosen Select(sigma_2_values)
    corr_coeff_chooser = @bind corr_coeff_chosen Select(corr_coeff_values)

    md"""
    ## What parameters to explore?

     ``\sigma_1= `` $(sigma_1_chooser)

     ``\sigma_2= `` $(sigma_2_chooser)

     ``\rho = `` $(corr_coeff_chooser)
    """

end

# ╔═╡ 02f93d8f-8a1c-4e0f-8a1a-dccde922b84e
begin
    filtered_df = df_monte[df_monte.Sigma_1.==sigma_1_chosen, :]
    filtered_df = filtered_df[filtered_df.Sigma_2.==sigma_2_chosen, :]
    filtered_df = filtered_df[filtered_df.Corr_Coeff.==corr_coeff_chosen, :]

    ratio_shown = 1
    filtered_df = filtered_df[filtered_df.Rel_Error.<=quantile(filtered_df.Rel_Error, ratio_shown), :]

    Plots.plot(100 .* filtered_df.Rel_Error, seriestype=:histogram, bins=length(filtered_df.Rel_Error), xlabel="Relative Error, %", ylabel="Frequency", title="Histogram of $(ratio_shown * 100)% Relative Errors. \$(\\sigma_1, \\sigma_2, \\rho) = ($(sigma_1_chosen), $(sigma_2_chosen), $(corr_coeff_chosen) )\$ ", size=(900, 600), label=false)

end

# ╔═╡ Cell order:
# ╟─ce1da846-d7f2-4cc2-bae7-f3446dbf7dec
# ╟─8bbd7875-665f-4949-87d0-a8a2d1ebdfb9
# ╠═c4f95e90-a3f7-11ee-179e-119f144e6845
# ╠═96ee9af6-0fd5-46f7-a68b-a00a1bd47b6c
# ╟─6286baf3-1124-4992-8779-c84e4a8b82dc
# ╠═1632b500-d0d9-47ca-be39-ad6b60d4817c
# ╠═9863370c-c52c-4906-9b75-720b475d38f3
# ╟─23e47540-5958-4ef3-9ae5-bc0b1222318d
# ╠═abd25bf2-ba31-4d9f-a674-b542e387824e
# ╠═c00895e5-480f-4aba-9474-4d4f6229be51
# ╠═ac952f41-be38-42a8-b2dc-95a244a6c5aa
# ╠═63de67ba-2f64-43c6-91f4-4f6164ebf5f9
# ╠═63b1567c-85cb-4def-a53d-48ba1ac15ced
# ╟─57adeb03-59ed-4e28-ab47-46c7209f9fd8
# ╟─5133d461-947f-4451-8478-a1cb0b9aef7d
# ╟─de7d650f-10a3-4b06-bd56-8e815d443d98
# ╟─80e65604-06ac-48bf-a792-8e042b471466
# ╠═0633345c-cf84-4b57-8d28-d7a513b0285a
# ╟─5c317ad2-7035-429e-ad67-8592d87c201c
# ╠═99a71d7f-5be8-4aa0-99f5-b588d57dae0e
# ╠═ead52d9e-3dad-4a15-919f-727f695f3529
# ╟─02f93d8f-8a1c-4e0f-8a1a-dccde922b84e
