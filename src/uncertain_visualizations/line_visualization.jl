
_is_line_parameter_stable(el) = 1e3 > abs(el) > 1e-3

"""
Scales the line by the parameter at the specified index.
Used for `ForwardDiff` package.
"""
function __scale_by_ind!(n_container, params, ind)
    s = params
    s = s ./ s[ind]
    s = deleteat(s, ind)
    n_container .= s
    return
end


"""
Returns vector n = (a / c, b / c)^T and the respective covariance matrix.

If division is unstable, the other parameter (`a` or `b`) is chosen for scaling.
"""
function _scale_the_line(l::UncertainLine, __param_ind=1)
    @assert __param_ind < 4 "A poor line was chosen. All the values are unstable."

    if !(_is_line_parameter_stable(l.params[__param_ind]))
        return _scale_the_line(l, __param_ind + 1)
    end

    n = zeros(2)
    closure_over_scaling!(n, args_vector) = __scale_by_ind!(n, args_vector, __param_ind)

    J = ForwardDiff.jacobian(closure_over_scaling!, n, l.params)
    n, propagate_the_covariance(J, l.covariance_matrix)
end


"""
Uses the same naming convention as in the `clarke.pdf` document (last section):
https://drive.google.com/file/d/1Q4BmXE510rrAn-3RnGGLmmjW9egWb9Vf/view?pli=1
"""
function plot_confidence_region(l::UncertainLine, confidence_level::Real, yrange, xrange)
    χ²_quantile = quantile(_χ²_2_DoF, confidence_level)

    n, Λₙ = _scale_the_line(l)

    Λ⁻¹ₙ = inv(Λₙ)

    Λ⁻¹_times_n = Λ⁻¹ₙ * n

    C = [
        Λ⁻¹ₙ[1, 1] Λ⁻¹ₙ[1, 2] -Λ⁻¹_times_n[1];
        Λ⁻¹ₙ[2, 1] Λ⁻¹ₙ[2, 2] -Λ⁻¹_times_n[2];
        -Λ⁻¹_times_n... (n'*Λ⁻¹_times_n-χ²_quantile)
    ]

    Cᵃ = inv(C)  # adjoint of C. up to scale: the true adjoint has to be multiplied by `det(C)`.

    quadratic_form(a, b) = dot([a, b, 1], Cᵃ * [a, b, 1])
    z = [quadratic_form(a, b) for a in yrange, b in xrange] # the convention is (y, x) as the points are stored this way.

    # PlotlyJS.plot(PlotlyJS.surface(x=xrange, y=yrange, z=z))   # DEBUG comment. To be removed.
    PlotlyJS.plot(PlotlyJS.contour(x=xrange, y=yrange, z=z; level=[0]))
end


# TODO: DEBUG
function DEBUG_line_visualization()
    l = find_uncertain_line(
        UncertainPoint(
            MVector{3}(9.0, 10.0, 1.0),
            MMatrix{3,3}(1.5 * [2.5 1.5 0;
                1.5 2 0;
                0 0 0
            ])
        ),
        UncertainPoint(
            MVector{3}(20.0, 22.0, 1.0),
            MMatrix{3,3}(1.5 * [3 1.5 0;
                1.5 2 0;
                0 0 0
            ])
        ))
    plot_confidence_region(l, 0.95, -15:0.05:30, -15:0.05:30)
end
