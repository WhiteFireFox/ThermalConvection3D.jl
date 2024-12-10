# ThermalConvection3D

## Physical Equations

### 1. Continuity Equation (Mass Conservation)

For incompressible flow, the mass conservation equation is:

$$
\nabla \cdot \mathbf{V} = 0
$$

where $\mathbf{V} = (V_x, V_y, V_z)$ is the velocity field.

### 2. Momentum Equations (Navier-Stokes Equations with Buoyancy)

The momentum equations, incorporating buoyancy via the Boussinesq approximation, are:

$$
\rho \left( \frac{\partial \mathbf{V}}{\partial t} + \mathbf{V} \cdot \nabla \mathbf{V} \right) = -\nabla P + \nabla \cdot \boldsymbol{\tau} + \mathbf{F}_{\text{buoyancy}}
$$

where:
- $\rho$ is the density.
- $P$ is the pressure.
- $\boldsymbol{\tau}$ is the deviatoric stress tensor.
- $\mathbf{F}_{\text{buoyancy}} = \rho_0 g \alpha (T - T_0) \mathbf{e}_z$ is the buoyancy force due to temperature variations ($T$ is temperature, $T_0$ is a reference temperature, $g$ is gravitational acceleration, $\alpha$ is the thermal expansion coefficient, and $\mathbf{e}_z$ is the unit vector in the vertical direction).

The deviatoric stress tensor for a Newtonian fluid is:

$$
\boldsymbol{\tau} = 2 \eta \mathbf{E}
$$

where $\eta$ is the dynamic (potentially temperature-dependent) viscosity, and $\mathbf{E}$ is the strain rate tensor:

$$
\mathbf{E} = \frac{1}{2} \left( \nabla \mathbf{V} + (\nabla \mathbf{V})^\top \right)
$$

### 3. Energy Equation (Heat Equation)

The energy equation accounting for advection and diffusion of heat is:

$$
\frac{\partial T}{\partial t} + \mathbf{V} \cdot \nabla T = \kappa \nabla^2 T
$$

where $\kappa$ is the thermal diffusivity.

## Pseudo-Transient Reformulation (Numerical Method)

To solve the above equations numerically, a pseudo-transient (or artificial compressibility) approach is often used to achieve a stable iterative scheme that can converge to steady-state or handle transient integration. This involves introducing pseudo-time steps for both the momentum and continuity equations, and using iterative solvers for the velocity–pressure coupling.

### 1. Momentum Equation with Pseudo-Time

A pseudo-time $\tau$ is introduced to iteratively solve for velocity fields

$$
\rho_0 \frac{\partial \mathbf{V}}{\partial \tau} + \rho_0 \left( \frac{\partial \mathbf{V}}{\partial t} + \mathbf{V} \cdot \nabla \mathbf{V} \right) = -\nabla P + \nabla \cdot \boldsymbol{\tau} + \mathbf{F}_{\text{buoyancy}}.
$$

By choosing $\frac{\partial \mathbf{V}}{\partial t}$ to represent the physical time derivative and $\frac{\partial \mathbf{V}}{\partial \tau}$ as a pseudo-time derivative, one can iterate in $\tau$ until a quasi-steady state in pseudo-time is reached for each real time step $t$.

### 2. Continuity with Artificial Compressibility

To circumvent solving a Poisson equation directly for pressure, an artificial compressibility $\beta$ can be introduced

$$
\beta \frac{\partial P}{\partial \tau} + \nabla \cdot \mathbf{V} = 0.
$$

As $\tau \to \infty$ (at each physical time step), $\nabla \cdot \mathbf{V} \to 0$, recovering incompressibility. The parameter $\beta$ is chosen to balance stability and convergence rates.

### 3. Temperature Update (Explicit/Implicit Time-Stepping)

Once the velocity field $\mathbf{u}$ is known at time $t$, we update the temperature field using an explicit time-stepping method. The discretized energy equation is

$$
T^{n+1} = T^{n} + \Delta t \left(\frac{dT}{dt}\right),
$$

where $\frac{dT}{dt}$ is computed from the spatial discretization of advection and diffusion terms

$$
\frac{dT}{dt} = -\mathbf{V}\cdot\nabla T + \kappa \nabla^2 T.
$$

However, for the temperature field, we also can use the implicit time-stepping approach with pseudo-transient method. At each physical time step $t$, we can introduce a pseudo-time $\tau$

$$
\frac{\partial T}{\partial \tau} + \frac{T - T_{\text{old}}}{dt} + \mathbf{V} \cdot \nabla T = \kappa \nabla^2 T,
$$

where $T_{\text{old}}$ is the temperature from the previous physical time step and $dt$ is the physical time step size.

## Result

we verify our code on a large scale (global grid of `476x156x156`) for 3D thermal porous convection with multi-xPUs (8 GPUs). The results are as follows.

```julia
ar         = 3                               # aspect ratio
nx, ny, nz = 80*ar-1, 80-1, 80-1             # numerical grid resolutions
nt         = 5000                            # total number of timesteps
nout       = 50                              # frequency of plotting
```

We can get the following results by executing this command.
```julia
sbatch ./scripts/sbatch.sh
```
or
```julia
srun -n8 bash -c "julia --project ./scripts/ThermalConvection3D.jl"
```

We provide a GIF showcasing the 2D slice of the 3D data at $ly/2$.

<div align="center">
    <img src="./docs/ThermalConvect2D_MPI.gif" width="50%">
</div>

We provide a file called `Visualise_3D.jl` to generate 3D image from `.bin` file. Then, we use `Visualise_GIF.py` to get a GIF showcasing the 3D data.

<div align="center">
    <img src="./docs/ThermalConvect3D_MPI.gif" width="50%">
</div>



