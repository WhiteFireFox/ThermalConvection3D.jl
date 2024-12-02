# ThermalConvection3D

## **Physical Equations**

### **1. Continuity Equation (Mass Conservation):**

For incompressible flow, the mass conservation equation is:

$$
\nabla \cdot \mathbf{V} = 0
$$

where $\mathbf{V} = (V_x, V_y)$ is the velocity field.

### **2. Momentum Equations (Navier-Stokes Equations with Buoyancy):**

The momentum equations, incorporating buoyancy via the Boussinesq approximation, are:

$$
\rho \left( \frac{\partial \mathbf{V}}{\partial t} + \mathbf{V} \cdot \nabla \mathbf{V} \right) = -\nabla P + \nabla \cdot \boldsymbol{\tau} + \mathbf{F}_{\text{buoyancy}}
$$

where:
- $\rho$ is the density.
- $P$ is the pressure.
- $\boldsymbol{\tau}$ is the deviatoric stress tensor.
- $\mathbf{F}_{\text{buoyancy}} = \rho_0 g \alpha (T - T_0) \mathbf{e}_y$ is the buoyancy force due to temperature variations ($T$ is temperature, $T_0$ is a reference temperature, $g$ is gravitational acceleration, $\alpha$ is the thermal expansion coefficient, and $\mathbf{e}_y$ is the unit vector in the vertical direction).

The deviatoric stress tensor for a Newtonian fluid is:

$$
\boldsymbol{\tau} = 2 \eta \mathbf{E}
$$

where $\eta$ is the dynamic viscosity, and $\mathbf{E}$ is the strain rate tensor:

$$
\mathbf{E} = \frac{1}{2} \left( \nabla \mathbf{V} + (\nabla \mathbf{V})^\top \right)
$$

### **3. Energy Equation (Heat Equation):**

The energy equation accounting for advection and diffusion of heat is:

$$
\frac{\partial T}{\partial t} + \mathbf{V} \cdot \nabla T = \kappa \nabla^2 T
$$

where $\kappa$ is the thermal diffusivity.