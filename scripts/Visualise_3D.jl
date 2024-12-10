using GLMakie

function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end

function visualise(file_path::String, output_path::String)
    lx, ly, lz = 3.0, 1.0, 1.0
    nx = 474
    ny = nz = 154
    T = zeros(Float32, nx, ny, nz)
    load_array(file_path, T)
    xc, yc, zc = LinRange(0, lx, nx), LinRange(0, ly, ny), LinRange(0, lz, nz)
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax = Axis3(fig[1, 1]; aspect=(1, 1, 0.5), title="Temperature", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_T = contour!(ax, xc, yc, zc, T; alpha=0.05, colormap=:turbo)
    save(output_path, fig)
    return fig
end

function batch_visualise(input_dir::String, output_dir::String, start_idx::Int, end_idx::Int)
    for i in start_idx:end_idx
        # Format the file index as a 4-digit number (e.g., "0001", "0002", etc.)
        file_name = string("out_T_", lpad(i, 4, '0'))
        input_path = joinpath(input_dir, file_name)
        output_path = joinpath(output_dir, "T_3D_$(lpad(i, 4, '0')).png")
        
        println("Processing: $input_path")
        visualise(input_path, output_path)
    end
end

batch_visualise("./viz3Dmpi_out", "./output", 0, 99)