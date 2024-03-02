module RansacDoneRight

using Revise, StaticArrays, Random, StatsBase, Plots, PlotlyJS, Colors, LinearAlgebra,
    Distributions, BlockDiagonals, DataFrames, Combinatorics, BenchmarkTools,
    ForwardDiff, Statistics

theme(:dracula::Symbol;)  # sets the theme for `Plotly` plots





# TODO: properly organize includes

include("correspondences/correspondences.jl")
include("correspondences/visualization.jl")

include("covariance_propagation/covariance_utils.jl")

include("solvers/homography_solvers/uncertain_homography.jl")
include("solvers/homography_solvers/better_faster_plane_measuring_device.jl")

include("normalizations/affine_normalization.jl")

include("residuals/uncertain_residuals.jl")
include("residuals/forward_residuals_computation.jl")
include("residuals/reprojection_residuals_computation.jl")


include("hypothesis_testing/uncertain_line.jl")
include("hypothesis_testing/flip_test.jl")
include("hypothesis_testing/correspondence_collinearity.jl")
include("hypothesis_testing/consensus_set_labelling.jl")
include("hypothesis_testing/ellipses_intersection_test.jl")


include("monte_carlo/reprojection_residuals_covariance_verification.jl")

include("normalizations/spherical_normalization.jl")



include("uncertain_visualizations/uncertain_point_visualization.jl")
include("uncertain_visualizations/line_visualization.jl")


end
