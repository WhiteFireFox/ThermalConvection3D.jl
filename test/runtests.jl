using Test
using ThermalConvection3D

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()

    printstyled("Testing ThermalConvection3D.jl\n"; bold=true, color=:white)

    run(`$exename -O3 --startup-file=no $(joinpath(testdir, "test3D.jl"))`)
    run(`$exename -O3 --startup-file=no $(joinpath(testdir, "test3D_implicit.jl"))`)

    return
end

runtests()
