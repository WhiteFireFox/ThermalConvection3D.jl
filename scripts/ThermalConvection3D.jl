const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots, Printf, Statistics, LinearAlgebra

@parallel function assign!(A::Data.Array, B::Data.Array)
    @all(A) = @all(B)
    return
end

@parallel function compute_error!(Err_A::Data.Array, A::Data.Array)
    @all(Err_A) = @all(Err_A) - @all(A)
    return
end

@parallel function compute_0!(RogT::Data.Array, Eta::Data.Array, ∇V::Data.Array, T::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, ρ0gα::Data.Number, η0::Data.Number, dη_dT::Data.Number, ΔT::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(RogT) = ρ0gα * @all(T)
    @all(Eta)  = η0 * (1.0 - dη_dT * (@all(T) + ΔT / 2.0))
    @all(∇V)   = @d_xa(Vx) / dx + @d_ya(Vy) / dy + @d_za(Vz) / dz
    return
end

@parallel function compute_1!(Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, σxy::Data.Array, σxz::Data.Array, σyz::Data.Array, Eta::Data.Array, ∇V::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dτ_iter::Data.Number, β::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(Pt)   = @all(Pt) - dτ_iter / β * @all(∇V)
    @all(τxx)  = 2.0 * @all(Eta) * (@d_xa(Vx) / dx - (1.0 / 3.0) * @all(∇V))
    @all(τyy)  = 2.0 * @all(Eta) * (@d_ya(Vy) / dy - (1.0 / 3.0) * @all(∇V))
    @all(τzz)  = 2.0 * @all(Eta) * (@d_za(Vz) / dz - (1.0 / 3.0) * @all(∇V))
    @all(σxy)  = 2.0 * @av(Eta) * (0.5 * (@d_yi(Vx) / dy + @d_xi(Vy) / dx))
    @all(σxz)  = 2.0 * @av(Eta) * (0.5 * (@d_zi(Vx) / dz + @d_xi(Vz) / dx))
    @all(σyz)  = 2.0 * @av(Eta) * (0.5 * (@d_zi(Vy) / dz + @d_yi(Vz) / dy))
    return
end


@parallel function compute_2!(Rx::Data.Array, Ry::Data.Array, Rz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, Pt::Data.Array, RogT::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, σxy::Data.Array, σxz::Data.Array, σyz::Data.Array, ρ::Data.Number, dampX::Data.Number, dampY::Data.Number, dampZ::Data.Number, dτ_iter::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(Rx)    = 1.0 / ρ * (@d_xi(τxx) / dx + @d_ya(σxy) / dy + @d_za(σxz) / dz - @d_xi(Pt) / dx)
    @all(Ry)    = 1.0 / ρ * (@d_yi(τyy) / dy + @d_xa(σxy) / dx + @d_za(σyz) / dz - @d_yi(Pt) / dy)
    @all(Rz)    = 1.0 / ρ * (@d_zi(τzz) / dz + @d_xa(σxz) / dx + @d_ya(σyz) / dy - @d_zi(Pt) / dz + @av_zi(RogT))
    @all(dVxdτ) = dampX * @all(dVxdτ) + @all(Rx) * dτ_iter
    @all(dVydτ) = dampY * @all(dVydτ) + @all(Ry) * dτ_iter
    @all(dVzdτ) = dampZ * @all(dVzdτ) + @all(Rz) * dτ_iter
    return
end

@parallel function update_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, dτ_iter::Data.Number)
    @inn(Vx) = @inn(Vx) + @all(dVxdτ) * dτ_iter
    @inn(Vy) = @inn(Vy) + @all(dVydτ) * dτ_iter
    @inn(Vz) = @inn(Vz) + @all(dVzdτ) * dτ_iter
    return
end

@parallel function compute_qT!(qTx::Data.Array, qTy::Data.Array, qTz::Data.Array, T::Data.Array, DcT::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(qTx) = -DcT * @d_xi(T) / dx
    @all(qTy) = -DcT * @d_yi(T) / dy
    @all(qTz) = -DcT * @d_zi(T) / dz
    return
end

@parallel_indices (ix, iy, iz) function advect_T!(dT_dt::Data.Array, qTx::Data.Array, qTy::Data.Array, qTz::Data.Array, T::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    if (ix <= size(dT_dt, 1) && iy <= size(dT_dt, 2) && iz <= size(dT_dt, 3))
        dT_dt[ix, iy, iz] = -((qTx[ix + 1, iy, iz] - qTx[ix, iy, iz]) / dx +
                            (qTy[ix, iy + 1, iz] - qTy[ix, iy, iz]) / dy +
                            (qTz[ix, iy, iz + 1] - qTz[ix, iy, iz]) / dz) -
                            (Vx[ix + 1, iy + 1, iz + 1] > 0) * Vx[ix + 1, iy + 1, iz + 1] * (T[ix + 1, iy + 1, iz + 1] - T[ix, iy + 1, iz + 1]) / dx -
                            (Vx[ix + 2, iy + 1, iz + 1] < 0) * Vx[ix + 2, iy + 1, iz + 1] * (T[ix + 2, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz + 1]) / dx -
                            (Vy[ix + 1, iy + 1, iz + 1] > 0) * Vy[ix + 1, iy + 1, iz + 1] * (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy, iz + 1]) / dy -
                            (Vy[ix + 1, iy + 2, iz + 1] < 0) * Vy[ix + 1, iy + 2, iz + 1] * (T[ix + 1, iy + 2, iz + 1] - T[ix + 1, iy + 1, iz + 1]) / dy -
                            (Vz[ix + 1, iy + 1, iz + 1] > 0) * Vz[ix + 1, iy + 1, iz + 1] * (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz]) / dz -
                            (Vz[ix + 1, iy + 1, iz + 2] < 0) * Vz[ix + 1, iy + 1, iz + 2] * (T[ix + 1, iy + 1, iz + 2] - T[ix + 1, iy + 1, iz + 1]) / dz
    end
    return
end

@parallel_indices (ix, iy, iz) function no_fluxZ_T!(T::Data.Array)
    nx, ny, nz = size(T)
    # x-direction
    if ix == 1 && iy <= ny && iz <= nz
        T[ix, iy, iz] = T[ix + 1, iy, iz]
    elseif ix == nx && iy <= ny && iz <= nz
        T[ix, iy, iz] = T[ix - 1, iy, iz]
    end

    # y-direction
    if iy == 1 && ix <= nx && iz <= nz
        T[ix, iy, iz] = T[ix, iy + 1, iz]
    elseif iy == ny && ix <= nx && iz <= nz
        T[ix, iy, iz] = T[ix, iy - 1, iz]
    end

    # # z-direction
    # if iz == 1 && ix <= nx && iy <= ny
    #     T[ix, iy, iz] = T[ix, iy, iz + 1]
    # elseif iz == nz && ix <= nx && iy <= ny
    #     T[ix, iy, iz] = T[ix, iy, iz - 1]
    # end
    return
end

@parallel function update_T!(T::Data.Array, T_old::Data.Array, dT_dt::Data.Array, dt::Data.Number)
    @inn(T) = @inn(T_old) + @all(dT_dt)*dt
    return
end

@parallel_indices (iy, iz) function bc_x!(A::Data.Array)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end - 1, iy, iz]
    return
end

@parallel_indices (ix, iz) function bc_y!(A::Data.Array)
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end - 1, iz]
    return
end

@parallel_indices (ix, iy) function bc_z!(A::Data.Array)
    A[ix, iy, 1  ] = A[ix, iy, 2    ]
    A[ix, iy, end] = A[ix, iy, end - 1]
    return
end

##################################################
@views function ThermalConvection3D()
    # Physics - dimensionally independent scales
    lz        = 1.0                # domain extent in z, m
    η0        = 1.0                # viscosity, Pa*s
    DcT       = 1.0                # heat diffusivity, m^2/s
    ΔT        = 1.0                # initial temperature perturbation K
    # Physics - nondim numbers
    Ra        = 1e7                # Rayleigh number = ρ0*g*α*ΔT*ly^3/η0/DcT
    Pra       = 1e3                # Prandtl number = η0/ρ0/DcT
    ar        = 3                  # aspect ratio
    # Physics - dimensionally dependent parameters
    ly        = lz
    lx        = ar * ly            # domain extent in x, m
    w         = 1e-2 * ly          # initial perturbation standard deviation, m
    ρ0gα      = Ra * η0 * DcT / ΔT / ly^3  # thermal expansion coefficient
    dη_dT     = 1e-10 / ΔT         # viscosity's temperature dependence
    # Numerics
    nx, ny, nz = 96 * ar - 1, 96 - 1, 96 - 1  # numerical grid resolutions
    iterMax   = 5 * 10^4            # maximal number of pseudo-transient iterations
    nt        = 3000                # total number of timesteps
    nout      = 10                  # frequency of plotting
    nerr      = 100                 # frequency of error checking
    ε         = 1e-4                # nonlinear absolute tolerance
    dmp       = 2                   # damping parameter
    st        = 5                   # quiver plotting spatial step
    # Derived numerics
    dx, dy, dz = lx / (nx - 1), ly / (ny - 1), lz / (nz - 1)  # cell sizes
    ρ         = 1.0 / Pra * η0 / DcT                # density
    dt_diff   = 1.0 / 6.1 * min(dx, dy, dz)^2 / DcT      # diffusive CFL timestep limiter
    dτ_iter   = 1.0 / 8.1 * min(dx, dy, dz) / sqrt(η0 / ρ)  # iterative CFL pseudo-timestep limiter
    β         = 8.1 * dτ_iter^2 / min(dx, dy, dz)^2 / ρ     # numerical bulk compressibility
    dampX     = 1.0 - dmp / nx                    # damping term for the x-momentum equation
    dampY     = 1.0 - dmp / ny                    # damping term for the y-momentum equation
    dampZ     = 1.0 - dmp / nz                    # damping term for the z-momentum equation
    # Array allocations
    T         = @zeros(nx, ny, nz)
    T        .= Data.Array([ΔT * exp(-(((ix - 1) * dx - 0.5 * lx) / w)^2 -
                                       (((iy - 1) * dy - 0.5 * ly) / w)^2 -
                                       (((iz - 1) * dz - 0.5 * lz) / w)^2)
                            for ix = 1:nx, iy = 1:ny, iz = 1:nz])
    T[:, :, 1  ] .=  ΔT / 2.0
    T[:, :, end] .= -ΔT / 2.0
    T_old     = @zeros(nx, ny, nz)
    Pt        = @zeros(nx, ny, nz)
    ∇V        = @zeros(nx, ny, nz)
    Vx        = @zeros(nx + 1, ny, nz)
    Vy        = @zeros(nx, ny + 1, nz)
    Vz        = @zeros(nx, ny, nz + 1)
    RogT      = @zeros(nx, ny, nz)
    Eta       = @zeros(nx, ny, nz)
    τxx       = @zeros(nx, ny, nz)
    τyy       = @zeros(nx, ny, nz)
    τzz       = @zeros(nx, ny, nz)
    σxy       = @zeros(nx - 1, ny - 1, nz)
    σxz       = @zeros(nx - 1, ny, nz - 1)
    σyz       = @zeros(nx, ny - 1, nz - 1)
    Rx        = @zeros(nx - 1, ny - 2, nz - 2)
    Ry        = @zeros(nx - 2, ny - 1, nz - 2)
    Rz        = @zeros(nx - 2, ny - 2, nz - 1)
    dVxdτ     = @zeros(nx - 1, ny - 2, nz - 2)
    dVydτ     = @zeros(nx - 2, ny - 1, nz - 2)
    dVzdτ     = @zeros(nx - 2, ny - 2, nz - 1)
    qTx       = @zeros(nx - 1, ny - 2, nz - 2)
    qTy       = @zeros(nx - 2, ny - 1, nz - 2)
    qTz       = @zeros(nx - 2, ny - 2, nz - 1)
    dT_dt     = @zeros(nx - 2, ny - 2, nz - 2)
    ErrP      = @zeros(nx, ny, nz)
    ErrVx     = @zeros(nx + 1, ny, nz)
    ErrVy     = @zeros(nx, ny + 1, nz)
    ErrVz     = @zeros(nx, ny, nz + 1)
    # Time loop
    err_evo1 = []; err_evo2 = []
    for it = 1:nt
        @parallel assign!(T_old, T)
        errVx, errVy, errVz, errP = 2 * ε, 2 * ε, 2 * ε, 2 * ε
        iter = 1; niter = 0
        while (maximum([errVx, errVy, errVz]) > ε || errP > ε) && iter <= iterMax
            # @parallel assign!(ErrVx, Vx)
            # @parallel assign!(ErrVy, Vy)
            @parallel assign!(ErrVz, Vz)
            @parallel assign!(ErrP, Pt)
            @parallel compute_0!(RogT, Eta, ∇V, T, Vx, Vy, Vz, ρ0gα, η0, dη_dT, ΔT, dx, dy, dz)
            @parallel compute_1!(Pt, τxx, τyy, τzz, σxy, σxz, σyz, Eta, ∇V, Vx, Vy, Vz, dτ_iter, β, dx, dy, dz)
            @parallel compute_2!(Rx, Ry, Rz, dVxdτ, dVydτ, dVzdτ, Pt, RogT, τxx, τyy, τzz, σxy, σxz, σyz, ρ, dampX, dampY, dampZ, dτ_iter, dx, dy, dz)
            @parallel update_V!(Vx, Vy, Vz, dVxdτ, dVydτ, dVzdτ, dτ_iter)
            @parallel (1:size(Vx, 2), 1:size(Vx, 3)) bc_x!(Vx)
            @parallel (1:size(Vy, 1), 1:size(Vy, 3)) bc_y!(Vy)
            @parallel (1:size(Vz, 1), 1:size(Vz, 2)) bc_z!(Vz)
            # @parallel compute_error!(ErrVx, Vx)
            # @parallel compute_error!(ErrVy, Vy)
            @parallel compute_error!(ErrVz, Vz)
            @parallel compute_error!(ErrP, Pt)
            if mod(iter, nerr) == 0
                # errVx = maximum(abs.(Array(ErrVx))) / (1e-12 + maximum(abs.(Array(Vx))))
                # errVy = maximum(abs.(Array(ErrVy))) / (1e-12 + maximum(abs.(Array(Vy))))
                errVz = maximum(abs.(Array(ErrVz))) / (1e-12 + maximum(abs.(Array(Vz))))
                errP  = maximum(abs.(Array(ErrP)))  / (1e-12 + maximum(abs.(Array(Pt))))
                push!(err_evo1, maximum([errVx, errVy, errVz, errP]))
                push!(err_evo2, iter)
                # @printf("Total steps = %d, errV=%1.3e, errP=%1.3e \n", iter, maximum([errVx, errVy, errVz]), errP)
            end
            iter += 1; niter += 1
        end
        # Thermal solver
        @parallel compute_qT!(qTx, qTy, qTz, T, DcT, dx, dy, dz)
        @parallel advect_T!(dT_dt, qTx, qTy, qTz, T, Vx, Vy, Vz, dx, dy, dz)
        dt_adv = min(dx / maximum(abs.(Array(Vx))),
                     dy / maximum(abs.(Array(Vy))),
                     dz / maximum(abs.(Array(Vz)))) / 3.1
        dt     = min(dt_diff, dt_adv)
        @parallel update_T!(T, T_old, dT_dt, dt)
        @parallel_indices (ix, iy, iz) no_flux_T!(T)
        @printf("it = %d (iter = %d), errV=%1.3e, errP=%1.3e \n", it, niter, maximum([errVx, errVy, errVz]), errP)
        # Visualization (optional)
        # if mod(it, nout) == 0
        #     # Example: plot a mid-plane slice
        #     iz_slice = nz ÷ 2
        #     heatmap(-lx / 2:dx:lx / 2, -ly / 2:dy:ly / 2, Array(T)[:, :, iz_slice]',
        #             aspect_ratio = 1, xlims = (-lx / 2, lx / 2), ylims = (-ly / 2, ly / 2),
        #             c = :inferno, clims = (-0.1, 0.1), title = "T° (it = $it of $nt, z = $(iz_slice * dz - lz / 2))")
        #     display(current())
        # end
    end
    return
end

ThermalConvection3D()
