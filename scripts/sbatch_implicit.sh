#!/bin/bash -l
#SBATCH --job-name="3D_thermal_convection_Implicit"
#SBATCH --output=./scripts/3D_thermal_convection_Implicit.out
#SBATCH --error=./scripts/3D_thermal_convection_Implicit.err
#SBATCH --time=20:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

srun -n8 bash -c "julia --project ./scripts/ThermalConvection3D_Implicit.jl"