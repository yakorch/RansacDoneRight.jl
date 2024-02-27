var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = RansacDoneRight","category":"page"},{"location":"#RansacDoneRight","page":"Home","title":"RansacDoneRight","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for RansacDoneRight.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [RansacDoneRight]","category":"page"},{"location":"#RansacDoneRight.Correspondence","page":"Home","title":"RansacDoneRight.Correspondence","text":"Correspondence{T<:Real} is a pair of points along with their covariances.\n\nRepresents the same feature from 2 different views.\n\n(x Σₓ)  (x Σₓ)\n\n\n\n\n\n","category":"type"},{"location":"#RansacDoneRight.UncertainHomography","page":"Home","title":"RansacDoneRight.UncertainHomography","text":"Represents an uncertain homography matrix H with covariance matrix Σₕ.\n\nHas 8 degrees of freedom and is full-rank. Marked mutable because can be normalized and hence modified.\n\n\n\n\n\n","category":"type"},{"location":"#RansacDoneRight.UncertainLine","page":"Home","title":"RansacDoneRight.UncertainLine","text":"Parameters of a line as l=(a, b, c)ᵀ.\n\nIf a homogeneous point x ∈ P³ lies on the line:\n\nx l = 0\n\n\n\n\n\n","category":"type"},{"location":"#RansacDoneRight.UncertainPoint","page":"Home","title":"RansacDoneRight.UncertainPoint","text":"The points are in homogeneous coordinates; covariance is 3×3.\n\n\n\n\n\n","category":"type"},{"location":"#RansacDoneRight.SmallBlockDiagonal!-Tuple{Any, Vararg{Any}}","page":"Home","title":"RansacDoneRight.SmallBlockDiagonal!","text":"Modifies B with blocks from matrices diagonally.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.SmallBlockDiagonal-Tuple{Any, Vararg{Any}}","page":"Home","title":"RansacDoneRight.SmallBlockDiagonal","text":"Does the same as BlockDiagonal module but is more efficient for small (length < 20) matrices.\n\nEvery matrix in matrices should be square.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.__scale_by_ind!-Tuple{Any, Any, Any}","page":"Home","title":"RansacDoneRight.__scale_by_ind!","text":"Scales the line by the parameter at the specified index. Used for ForwardDiff package.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._compute_H!-Union{Tuple{Arr}, Tuple{Any, Arr}} where Arr<:(AbstractVector{<:Real})","page":"Home","title":"RansacDoneRight._compute_H!","text":"Takes non-degenerate correspondences. Updates H_container. H - homography.\n\nminimal_set is flattened with function correspondences._get_minimal_set_flattened:\n\na₁ b₁ c₁ d₁ a₂  a₄ b₄ c₄ d₄\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._compute_H!-Union{Tuple{T}, Tuple{Any, AbstractMatrix{T}, AbstractMatrix{T}}} where T<:Real","page":"Home","title":"RansacDoneRight._compute_H!","text":"U and V are matrices of size 3×4, with correspondences in the columns.\n\nU stands for correspondences in the first image.\nV stands for correspondences in the second image.\n\nEliminates 1 degree of freedom by enforcing λ₄=1.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._compute_H_and_J_H-Tuple{StaticArraysCore.MVector{4, RansacDoneRight.Correspondence{Float64}}}","page":"Home","title":"RansacDoneRight._compute_H_and_J_H","text":"Takes the minimal set of correspondences, returns (H, Jₕ)\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._compute_residual!-Union{Tuple{T}, Tuple{StaticArraysCore.MVector{4, T}, Any, Union{LinearAlgebra.LU, StaticArrays.LU}, StaticArraysCore.SVector{2, T}, StaticArraysCore.SVector{2, T}}} where T<:Real","page":"Home","title":"RansacDoneRight._compute_residual!","text":"Saves the residual in r_container ∈ R⁴.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._compute_residual-Union{Tuple{T}, Tuple{Any, Union{LinearAlgebra.LU, StaticArrays.LU}, StaticArraysCore.SVector{2, T}, StaticArraysCore.SVector{2, T}}} where T<:Real","page":"Home","title":"RansacDoneRight._compute_residual","text":"Residual - 2 stacked distance vectors. Includes the symmetric error.\n\nu and v ∈ R² – points on the first and the second image, respectively.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._compute_residual_autodiff_wrapper!-Union{Tuple{T}, Tuple{StaticArraysCore.MVector{4, T}, StaticArraysCore.MVector{13, T}}} where T<:Real","page":"Home","title":"RansacDoneRight._compute_residual_autodiff_wrapper!","text":"A wrapper for AutoDiff pkg to work.\n\nh_and_points=vec(H) a b c d\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._generate_random_correspondence-Tuple{Any, Any}","page":"Home","title":"RansacDoneRight._generate_random_correspondence","text":"_generate_random_correspondence(1.0 * I(3), [800.0, 1200.0])\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._get_covariance_matrix_of_minimal_set-Tuple{V} where V<:AbstractVector{RansacDoneRight.Correspondence{Float64}}","page":"Home","title":"RansacDoneRight._get_covariance_matrix_of_minimal_set","text":"Minimal set ⟹ Σₓ, where Σₓ is block-diagonal and of shape 16×16, where each block is 2×2.\n\nHas the same order of covariances blocks as in the function _get_minimal_set_flattened.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._get_minimal_set_flattened-Tuple{StaticArraysCore.MVector{4, RansacDoneRight.Correspondence{Float64}}}","page":"Home","title":"RansacDoneRight._get_minimal_set_flattened","text":"Let minimal_set = Correspondence(a b 1 Σ₁ c d 1 Σ₂) .\n\nThen the correspondences are flattened as:\n\na₁ b₁ c₁ d₁ a₂  a₄ b₄ c₄ d₄\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight._scale_the_line","page":"Home","title":"RansacDoneRight._scale_the_line","text":"Returns vector n = (a / c, b / c)^T and the respective covariance matrix.\n\nIf division is unstable, the other parameter (a or b) is chosen for scaling.\n\n\n\n\n\n","category":"function"},{"location":"#RansacDoneRight.add_noise-Union{Tuple{M}, Tuple{V}, Tuple{V, M, M, Bool}} where {V<:AbstractVector{RansacDoneRight.Correspondence{Float64}}, M<:AbstractMatrix{Float64}}","page":"Home","title":"RansacDoneRight.add_noise","text":"add_noise(corresps, Σ₁, Σ₂, true)\n\nif actually_add_noise is false, just updates the covariance matrices.\n\nnoise_mᵢ is 2 × 2 and is the noise covariance matrix of the points at image i;\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.are_three_points_collinear-Tuple{RansacDoneRight.UncertainPoint{Float64}, RansacDoneRight.UncertainPoint{Float64}, RansacDoneRight.UncertainPoint{Float64}, Float64, Float64}","page":"Home","title":"RansacDoneRight.are_three_points_collinear","text":"T₁ and T₂ are the same as in function verify_no_points_are_collinear.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.compute_homography-Tuple{StaticArraysCore.MVector{4, RansacDoneRight.Correspondence{Float64}}}","page":"Home","title":"RansacDoneRight.compute_homography","text":"Returns the homography H only.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.compute_identity_statistic-Tuple{RansacDoneRight.UncertainPoint{Float64}, RansacDoneRight.UncertainPoint{Float64}}","page":"Home","title":"RansacDoneRight.compute_identity_statistic","text":"Computes the statistic of 2 image points identity.\n\nFollows a Chi-squared distribution with 2 deg. of freedom (only considers x and y coordinates.)\n\nTwo points are identical (p₁ ≡ p₂) when the statistic is close to 0.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.compute_incidence_statistic-Tuple{RansacDoneRight.UncertainPoint{Float64}, RansacDoneRight.UncertainLine{Float64}}","page":"Home","title":"RansacDoneRight.compute_incidence_statistic","text":"Computes the statistic of incidence of the image point and the line. Assumes the independence of the point and the line.\n\nFollows a Chi-squared distribution with 1 deg. of freedom.\n\np ∈ l ~ the statistic is close to 0.\n\nUses the conventions introduced by Forstner: https://www.ipb.uni-bonn.de/pdfs/Forstner2004Uncertainty.pdf\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.compute_inlier_mask-Union{Tuple{V}, Tuple{V, Float64}} where V<:AbstractVector{RansacDoneRight.UncertainResidual{Float64}}","page":"Home","title":"RansacDoneRight.compute_inlier_mask","text":"Performs the following test:\n\nH₀: Correspondence is an inlier H₁: Correspondence is an outlier.\n\nmask contains true where H₀ wasn't rejected.\n\nT is the test statistic corresponding to the quantile of the Chi-squared distribution with 4 deg. of freedom. 9.49 corresponds to a confidence level of 0.95.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.compute_inlier_test_statistic-Tuple{RansacDoneRight.UncertainResidual{Float64}}","page":"Home","title":"RansacDoneRight.compute_inlier_test_statistic","text":"compute_inlier_test_statistic(UncertainResidual{Float64})\n\nUncertain residual is assumed to have a mean at 0 and follow a Multivariate Gaussian distribution.\n\nCalculates the Mahalanobis distance which follows a Chi-squared distribution with 4 deg. of freedom.\n\nOne may use Distributions.quantile(_χ²_4_DoF, confidence_level) to get the threshold for the test.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.compute_uncertain_homography-Tuple{StaticArraysCore.MVector{4, RansacDoneRight.Correspondence{Float64}}}","page":"Home","title":"RansacDoneRight.compute_uncertain_homography","text":"get_H_and_covariance(MinimalSetCorrespondences)\n\nTakes minimal set and returns an uncertain homography.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.compute_uncertain_residuals-Union{Tuple{V}, Tuple{RansacDoneRight.UncertainHomography{Float64}, V}} where V<:AbstractVector{RansacDoneRight.Correspondence{Float64}}","page":"Home","title":"RansacDoneRight.compute_uncertain_residuals","text":"Returns the residuals along with respective covariances.\n\ncorrespondences and uncertain H are assumed to be independent.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.find_uncertain_line-Tuple{RansacDoneRight.UncertainPoint{Float64}, RansacDoneRight.UncertainPoint{Float64}}","page":"Home","title":"RansacDoneRight.find_uncertain_line","text":"Assumes two points are independent.\n\nUses the conventions introduced by Forstner: https://www.ipb.uni-bonn.de/pdfs/Forstner2004Uncertainty.pdf page 23\n\nDoes not perform spherical normalization.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.generate_k_random_correspondences","page":"Home","title":"RansacDoneRight.generate_k_random_correspondences","text":"generate_k_random_correspondences(k, H, [y, x])\n\nGenerates k random correspondences inside an image.\n\nExamples\n\njulia> using LinearAlgebra\njulia> generate_k_random_correspondences(5, I(3), [1080, 1920])\n        <5 correspondences>\n\n\n\n\n\n","category":"function"},{"location":"#RansacDoneRight.generate_random_minimal_set","page":"Home","title":"RansacDoneRight.generate_random_minimal_set","text":"Samples 4 uniform points within the image bounds.\n\n\n\n\n\n","category":"function"},{"location":"#RansacDoneRight.get_skew_symmetric_matrix_for_cross_product-Tuple{StaticArraysCore.MVector{3, Float64}}","page":"Home","title":"RansacDoneRight.get_skew_symmetric_matrix_for_cross_product","text":"For an expression of type\n\nc = x  y\n\nreturns the skew-symmetric matrix Sₓ s.t. \n\nc = Sₓy\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.is_minimal_set_flipped-Tuple{StaticArraysCore.MVector{4, RansacDoneRight.Correspondence{Float64}}}","page":"Home","title":"RansacDoneRight.is_minimal_set_flipped","text":"Tests whether the minimal set has been flipped.\n\nUses the approach proposed by RH. 3 correspondences is enough for the test.\n\nIf the minimal set isn't flipped, the correspondences are valid.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.monte_carlo_cov_residuals-Union{Tuple{V}, Tuple{V, Integer, Any}, Tuple{V, Integer, Any, Any}} where V<:AbstractVector{RansacDoneRight.Correspondence{Float64}}","page":"Home","title":"RansacDoneRight.monte_carlo_cov_residuals","text":"Runs the Monte Carlo to estimate the accuracy of First-Order Error Propagation for the covariance of the residuals.\n\nArguments:\n\n- `truth_correspondences` -- unnoised correspondences\n- `run_each_corresp` -- number of runs\n- `noise_matrix` -- the 2 × 2 covariance matrix of the points' noise.\n- `verbose` -- if `true`, displays large residuals when encountered and adds extra prints for Debug.\n\nParallelized version (uses Threads).\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.normalize_onto_unit_sphere","page":"Home","title":"RansacDoneRight.normalize_onto_unit_sphere","text":"Scales the uncertain (random) entity onto the unit sphere. Usually used for homogeneous entities.\n\nIf length(elements) = N, the covariance_matrix must be N × N.\n\nReturns normalized elements and the approximation of the covariance matrix.\n\n\n\n\n\n","category":"function"},{"location":"#RansacDoneRight.plot_confidence_region-Tuple{RansacDoneRight.UncertainLine, Real, Any, Any}","page":"Home","title":"RansacDoneRight.plot_confidence_region","text":"Uses the same naming convention as in the clarke.pdf document (last section): https://drive.google.com/file/d/1Q4BmXE510rrAn-3RnGGLmmjW9egWb9Vf/view?pli=1\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.plot_confidence_region-Tuple{RansacDoneRight.UncertainPoint{Float64}, Real}","page":"Home","title":"RansacDoneRight.plot_confidence_region","text":"plot_confidence_region(UncertainPoint{Float64}, 0.95)\n\nThe last component of p is assumed to be constant 1.\n\nconfidence_level = 1 - α\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.plot_correspondences-Tuple{V} where V<:AbstractVector{RansacDoneRight.Correspondence{Float64}}","page":"Home","title":"RansacDoneRight.plot_correspondences","text":"Plots 2-view 2D scene.\n\nimage_bounds assumed to be (y, x).\n\nerrors is an error of the point: could represent a relative error from the Monte Carlo. Should be in the same order as correspondences vector.\n\nIf no errors provided, doesn't plot the errors.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.populate_outliers!","page":"Home","title":"RansacDoneRight.populate_outliers!","text":"Populates the outliers with random points within the image bounds.\n\n\n\n\n\n","category":"function"},{"location":"#RansacDoneRight.propagate_the_covariance-Tuple{Any, Any}","page":"Home","title":"RansacDoneRight.propagate_the_covariance","text":"propagate_the_covariance(J, Σₓ) = Σ_f\n\nFor expressions of type Y = f(X), where the covariance of X (Σₓ) is known.\n\nΣ_f  J  Σₓ  Jᵀ\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.run_monte_carlo_comparison-Tuple{}","page":"Home","title":"RansacDoneRight.run_monte_carlo_comparison","text":"Runs Monte Carlo a lot of times and returns a DataFrame with results\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.swap_points_in_correspondence!-Tuple{RansacDoneRight.Correspondence{Float64}}","page":"Home","title":"RansacDoneRight.swap_points_in_correspondence!","text":"swap_points_in_correspondence!({x ⟺ x'})\n\nPoints in image 1 (x) become points in image 2 (x'), and vice versa.\n\n\n\n\n\n","category":"method"},{"location":"#RansacDoneRight.verify_no_points_are_collinear-Tuple{StaticArraysCore.MVector{4, RansacDoneRight.Correspondence{Float64}}, Float64, Float64, Bool}","page":"Home","title":"RansacDoneRight.verify_no_points_are_collinear","text":"Returns true if no points in the minimal set are collinear, and false otherwise.\n\nT₁ is the threshold statistic used for 2 points indentity test. Is some quantile(χ²₂). For example T₁=5.99 corresponds to confidence level of 95%.\nT₂ is the threshold statistic used for point-line incidence test. Is some quantile(χ²₁). For example T₂=3.84 corresponds to confidence level of 95%.\ncheck_flip tells whether to check for the flip between the correspondences.\n\n\n\n\n\n","category":"method"}]
}
