# Finite Element Analysis of a Rectangular Block with a Circular Hole

Kindy refer to the attached project PDF (Project Study_FiniteElementAnalysis.pdf) to understand more about Finite Element Analysis.

## Overview
This project applies the Finite Element Method (FEM) to analyze the stress and displacement distribution in a rectangular block with a circular hole. The study compares results from a Python-based FEM solver with those from commercial FEA software (Altair HyperWorks).

## Features
- **Python-based FEM solver**: Assembles global stiffness matrix and solves for displacements.
- **Mesh Convergence Study**: Demonstrates the effect of mesh refinement on solution accuracy.
- **Validation with Altair HyperWorks**: Compares results with a professional FEA tool.
- **Boundary Condition Implementation**: Ensures structural stability.

## Repository Structure
- `Code/` → Python script implementing the FEM solution.
- `Results/` → Visualizations and comparison of FEM solutions.
- `Meshes/` → Different mesh configurations used in the study.
- `Report.pdf` → Detailed explanation of the methodology and results.

## Getting Started
### Dependencies
Ensure you have the following installed:
- Python 3.x
- NumPy
- Matplotlib (for visualizations)

### Running the Code
1. Clone the repository:
   ```sh
   git clone https://github.com/NaveenJagadeesan/FEM-Project---Rectangular-Block-with-Discontinuity.git
   cd FEM-Project---Rectangular-Block-with-Discontinuity/Code
   ```
2. Run the Python script:
   ```sh
   python RectangularBlockWithHole.py
   ```

## Results
The project demonstrates that FEM accurately predicts deformation and stress concentration around the hole. The Python code results closely match the Altair HyperWorks simulation, confirming the validity of the methodology.

## Author
[Naveen Jagadeesan](https://github.com/NaveenJagadeesan)
