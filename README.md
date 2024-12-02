# ThermalConvection3D
Here are the equations formatted in Markdown:

## Physical Equations

**Mass Conservation Equation**

$
\nabla \cdot \mathbf{V} = 0
$
where $\mathbf{V} = (V_x, V_y)$ is the velocity field.

**Momentum Equations**

$
\rho \left( \frac{\partial \mathbf{V}}{\partial t} + \mathbf{V} \cdot \nabla \mathbf{V} \right) = -\nabla P + \nabla \cdot \boldsymbol{\tau} + \mathbf{F}_{\text{buoyancy}}
$
where:
- $\rho$ is the density,
- $P$ is the pressure,
- $\boldsymbol{\tau}$ is the deviatoric stress tensor,
- $\mathbf{F}_{\text{buoyancy}} = \rho_0 g \alpha (T - T_0) \mathbf{e}_y$ is the buoyancy force due to temperature variations.

Here:
- $T$ is the temperature,
- $T_0$ is a reference temperature,
- $g$ is the gravitational acceleration,
- $\alpha$ is the thermal expansion coefficient,
- $\mathbf{e}_y$ is the unit vector in the vertical direction.

**Deviatoric Stress Tensor**

For a Newtonian fluid:
$
\boldsymbol{\tau} = 2 \eta \mathbf{E}
$
where:
- $\eta$ is the dynamic viscosity,
- $\mathbf{E}$ is the strain rate tensor, defined as:
$
\mathbf{E} = \frac{1}{2} \left( \nabla \mathbf{V} + (\nabla \mathbf{V})^\top \right)
$

**Energy Equation**

$
\frac{\partial T}{\partial t} + \mathbf{V} \cdot \nabla T = \kappa \nabla^2 T
$
where $\kappa$ is the thermal diffusivity.