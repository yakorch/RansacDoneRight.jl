### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ b48ceec7-c617-4fc0-8bca-92929faf6c50
begin
	using Revise
	using StaticArrays
	using StatsBase
	using LinearAlgebra
	using Combinatorics
	using Plots
	using Statistics

	using PyCall
	using Distributions

	md"""## Package Imports"""
end

# ╔═╡ 17e83446-d245-11ee-376e-2d07bba971c4
begin
    using Pkg
	NOTEBOOKS_PATH = "src/pluto_notebooks"
	DATA_PATH = "data"  # store generated by notebook data
    @assert endswith(pwd(), NOTEBOOKS_PATH) "Wrong directory!"
	RDR_PATH = "../.."  # path to the RDR project root
    Pkg.activate(RDR_PATH)
	
	import RansacDoneRight as RDR
    md"""
    ### Activating RansacDoneRight environment...
    """
end

# ╔═╡ b59b47e0-3c1e-4298-8bee-c0ec2ecf767c
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2400px;
		padding-left: max(160px, 10%);
		padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 169da2fc-586b-4433-abe4-c80781175dbb
"""
	synthetic_labelling_test(I(2), 2. * I(2), 5.99, 3.84, 9.49, 0.4, 1_000, 1_000)

Performs synthetic experiments to verify the viability of probabilistic approach for consensus set labelling and covariance propagation.

- `noise_mᵢ` is the 2x2 covariance matrix of the noise to be applied to the points in the `i`-th image.

- `T₁` is the threshold  statistic for points identity test. Set `0.0` if no check for identity is needed.
- `T₂` is the threshold statistic for point-line incidence test. Set `0.0` if no check for incidence is needed.
- `T₃` is the threshold statistic for consensus set labelling. See `RDR.compute_inlier_mask` for more details.

- `outlier_ratio` is the ratio of outliers to be present in all correspondences. For example, if `0.3` is passed, `30%` of correspondences will be corrupted.

- `n_corresps` is the total number of correspondences to be generated.
- `n_runs` is the number of simulations to be run.


The points flip check is not being performed.


"""
function synthetic_labelling_test(noise_m₁, noise_m₂, T₁::Float64, T₂::Float64, T₃::Float64, outlier_ratio::Float64, n_corresps::Integer, n_runs::Integer)
	
	@assert T₁ >= 0. "Can't have a negative threshold statistic"
	@assert T₂ >= 0. "Can't have a negative threshold statistic"
	@assert 0. <= outlier_ratio < 1. "An ivalid outlier ratio"
	
	image_bounds = [800., 1200.]

	n_outliers = Integer(round(n_corresps * outlier_ratio)) + 1  # + 1 to not introduce errors with 1:0 UnitRange. 
	
	outlier_indices = 1:n_outliers

	confusion_matrix_labelling = zeros(2, 2)
	
	apt_noise_m₁ = 1. * noise_m₁
	apt_noise_m₂ = 1. * noise_m₂

	inlier_indices = (n_outliers+1):n_corresps
	
	gt_inlier_mask = trues(n_corresps)
	gt_inlier_mask[1:n_outliers] .= false

	ForwardStatistics = Float64[]
	TwoWayStatistics = Float64[]
	WholesomeStatistics = Float64[]
	ReprojectionStatistics = Float64[]
	
	UniformStatistics = Float64[]
	uniform_variance = det(apt_noise_m₁) ^ 0.5
	
	CovRANSACStatistics = Float64[]
	
	TrueLabels = Float64[]

	for _ in 1:n_runs
		certain_minimal_set = RDR.generate_random_minimal_set(image_bounds)
		certain_H = RDR.compute_homography(certain_minimal_set)

		corresps = RDR.generate_k_random_correspondences(n_corresps, certain_H, image_bounds)
		RDR.populate_outliers!((@view corresps[outlier_indices]), image_bounds)

		noised_corresps = RDR.add_noise(corresps, apt_noise_m₁, apt_noise_m₂, true)

		minimal_set_indices::MVector{4, Integer} = sample(inlier_indices, 4, replace=false)
		non_minimal_set_indices = setdiff(1:n_corresps, minimal_set_indices)

		minimal_set::MVector{4, RDR.Correspondence{Float64}} = @view noised_corresps[minimal_set_indices]

		minimal_set_is_viable = RDR.verify_no_points_are_collinear(minimal_set, T₁, T₂, true)

		if !minimal_set_is_viable
			continue
		end

		uncertain_homography = RDR.compute_uncertain_homography(minimal_set)
		RDR.swap_points_in_correspondence!.(minimal_set)
		uncertain_homography_inv = RDR.compute_uncertain_homography(minimal_set)

		reprojection_residuals = RDR.compute_uncertain_reprojection_residuals(uncertain_homography, @view noised_corresps[non_minimal_set_indices])
		forward_residuals = RDR.compute_uncertain_forward_residuals(uncertain_homography, @view noised_corresps[non_minimal_set_indices])
		
		RDR.swap_points_in_correspondence!.(@view noised_corresps[non_minimal_set_indices])
		backward_residuals = RDR.compute_uncertain_forward_residuals(uncertain_homography_inv, @view noised_corresps[non_minimal_set_indices])

		forward_statistics = RDR.compute_inlier_test_statistic.(forward_residuals)

		backward_statistics = RDR.compute_inlier_test_statistic.(backward_residuals)
		two_way_statistics = max.(forward_statistics, backward_statistics)
		wholesome_statistics = min.(forward_statistics, backward_statistics)

		reprojection_statistics = RDR.compute_inlier_test_statistic.(reprojection_residuals)
		
		d²_statistics = @. RDR.squared_norm(forward_residuals) / uniform_variance

		ellipses_tangency_statistics = RDR.compute_CovRANSAC_labelling_statistics(uncertain_homography, @view noised_corresps[non_minimal_set_indices])
		

		append!(ForwardStatistics, forward_statistics)
		append!(TwoWayStatistics, two_way_statistics)
		append!(WholesomeStatistics, wholesome_statistics)
		append!(ReprojectionStatistics, reprojection_statistics)
		append!(UniformStatistics, d²_statistics)
		append!(CovRANSACStatistics, ellipses_tangency_statistics)
		
		predicted_inlier_mask_RDR = two_way_statistics .< T₃
		for j in 1:(n_corresps-4)
			confusion_matrix_labelling[(j > n_outliers) + 1, predicted_inlier_mask_RDR[j] + 1] += 1
		end
		
		append!(TrueLabels, @view gt_inlier_mask[non_minimal_set_indices])
	end
	return confusion_matrix_labelling, [ForwardStatistics, TwoWayStatistics, WholesomeStatistics, ReprojectionStatistics, UniformStatistics, CovRANSACStatistics], TrueLabels
end

# ╔═╡ 26827270-dd23-46f5-a72e-3462d36f65ae
begin
	total_runs = 2_000

	point_identity_alpha = 0.01
	point_identity_st = quantile(RDR._χ²_2_DoF, 1 - point_identity_alpha)

	point_line_incidence_alpha = 0.01
	point_line_incidence_st = quantile(RDR._χ²_1_DoF, 1 - point_line_incidence_alpha)

	labelling_alpha = 0.01
	labelling_st = quantile(RDR._χ²_2_DoF, 1 - labelling_alpha)

	conf_m_labelling, raw_statistics, TrueLabels = synthetic_labelling_test(
		[1.2 -0.9; -0.9 2.5],
		[1.8 0.3; 0.3 1.7],
		point_identity_st,
		point_line_incidence_st,
		labelling_st,
		0.3,
		1000,
		total_runs)
	
	labels = ["Forward", "Two Way | Max", "Two Way | Min", "Reprojection", "Uniform Thresholding", "Cov-RANSAC"]

	conf_m_labelling_percents = conf_m_labelling
	for j in 1:2
		conf_m_labelling_percents[j, :] /= (sum(conf_m_labelling_percents[j, :]) / 100)
	end

	md"""
	##### Choose the simulation parameters here.
	"""
end

# ╔═╡ dfb98830-98d4-4876-bdaf-a25fa43bebd4
md"""
#### Consensus Set Labelling Analysis.

Specified significance level (`α`) for:
1. Points Identity Test: **$(point_identity_alpha)**
2. Point-Line Incidence Test: **$(point_line_incidence_alpha)**
3. Consensus set labelling: **$(labelling_alpha)**
_______________________________________________________________________________________________________________________________________________
- Empirical significance level (`α`): **$(round(conf_m_labelling_percents[2, 1]; digits=5))** %
- Empirical power of the test (`1-β`): **$(round(conf_m_labelling_percents[1, 1]; digits=5))** %
"""

# ╔═╡ 8fa4826d-885e-43eb-9eb0-6b86b73e5f2f
begin
	py"""
	import sys
	sys.path.append('../hypothesis_testing')
	"""
	roc_plots = pyimport("roc_analysis")
	md"""
	#### Python is set up!
	"""
end

# ╔═╡ 2fbd1427-231d-4b46-a4fa-6ca410bc13e5
begin
	viable = trues(length(TrueLabels))
	for stats in raw_statistics
		viable .&= isfinite.(stats)
	end

	cleaned_statistics = [stats[viable] for stats in raw_statistics]
	CleanLabels = TrueLabels[viable]
	
	roc_figure_path = DATA_PATH * "/" * "ROC_Curves_analysis.pdf"
	roc_plots.plot_roc_curves(
		cleaned_statistics,
		labels,
		CleanLabels,
		roc_figure_path)
end

# ╔═╡ Cell order:
# ╟─b59b47e0-3c1e-4298-8bee-c0ec2ecf767c
# ╠═17e83446-d245-11ee-376e-2d07bba971c4
# ╟─b48ceec7-c617-4fc0-8bca-92929faf6c50
# ╠═169da2fc-586b-4433-abe4-c80781175dbb
# ╠═26827270-dd23-46f5-a72e-3462d36f65ae
# ╟─dfb98830-98d4-4876-bdaf-a25fa43bebd4
# ╠═8fa4826d-885e-43eb-9eb0-6b86b73e5f2f
# ╠═2fbd1427-231d-4b46-a4fa-6ca410bc13e5
