# ThermalConvection3D



## Physical Equations

For incompressible flow, the mass conservation equation is:

∇⋅V=0\nabla \cdot \mathbf{V} = 0

where V=(Vx,Vy)\mathbf{V} = (V_x, V_y) is the velocity field.



The momentum equations, incorporating buoyancy via the Boussinesq approximation, are:

ρ(∂V∂t+V⋅∇V)=−∇P+∇⋅τ+Fbuoyancy\rho \left( \frac{\partial \mathbf{V}}{\partial t} + \mathbf{V} \cdot \nabla \mathbf{V} \right) = -\nabla P + \nabla \cdot \boldsymbol{\tau} + \mathbf{F}_{\text{buoyancy}}

where:

- ρ\rho is the density.
- PP is the pressure.
- τ\boldsymbol{\tau} is the deviatoric stress tensor.
- Fbuoyancy=ρ0gα(T−T0)ey\mathbf{F}_{\text{buoyancy}} = \rho_0 g \alpha (T - T_0) \mathbf{e}_y is the buoyancy force due to temperature variations (TT is temperature, T0T_0 is a reference temperature, gg is gravitational acceleration, α\alpha is the thermal expansion coefficient, and ey\mathbf{e}_y is the unit vector in the vertical direction).

The deviatoric stress tensor for a Newtonian fluid is:

τ=2ηE\boldsymbol{\tau} = 2 \eta \mathbf{E}

where η\eta is the dynamic viscosity, and E\mathbf{E} is the strain rate tensor:

E=12(∇V+(∇V)⊤)\mathbf{E} = \frac{1}{2} \left( \nabla \mathbf{V} + (\nabla \mathbf{V})^\top \right)



The energy equation accounting for advection and diffusion of heat is:

∂T∂t+V⋅∇T=κ∇2T\frac{\partial T}{\partial t} + \mathbf{V} \cdot \nabla T = \kappa \nabla^2 T

where κ\kappa is the thermal diffusivity.

