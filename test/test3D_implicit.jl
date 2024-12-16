using Test

include("../scripts/ThermalConvection3D_Implicit.jl")

function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end

@testset "Unit Test" begin
    lx,ly,lz    = 4.0, 2.0, 2.0
    nx,ny,nz    = 4, 2, 2
    dx,dy,dz    = lx/nx,ly/ny,lz/nz

    η0          = 1.0
    DcT         = 1.0
    ΔT          = 1.0
    Ra          = 1e7
    ρ0gα        = Ra * η0 * DcT / ΔT / ly^3
    dη_dT       = 1e-10 / ΔT

    T           = @rand(nx, ny, nz)
    RogT        = @rand(nx, ny, nz)
    Eta         = @rand(nx, ny, nz)
    ∇V          = @rand(nx, ny, nz)
    Vx          = @rand(nx + 1, ny, nz)
    Vy          = @rand(nx, ny + 1, nz)
    Vz          = @rand(nx, ny, nz + 1)

    ∇V_test     = copy(∇V)

    @parallel compute_0!(RogT, Eta, ∇V, T, Vx, Vy, Vz, ρ0gα, η0, dη_dT, ΔT, dx, dy, dz)
    ∇V_test     = diff(Vx, dims=1)./ dx .+ diff(Vy, dims=2)./ dy .+ diff(Vz, dims=3)./ dz

    @test ∇V    ≈ ∇V_test
end

@testset "Reference Test" begin
    T           = ThermalConvection3D()
    T_ref       = zeros(Float32, 45, 13, 13)
    load_array("./bin/out_T_implicit", T_ref)
    @test T_ref ≈ T
end

