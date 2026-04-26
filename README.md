# README

## Project: Restricted Three-Body Problem (Numerical Simulation)

This project contains Python implementations for solving and analyzing the Restricted Three-Body Problem.

The code is organized according to Programming Tasks (PT) and Questions (Q) in the assignment.


## Programming Language
Python 3


## Required Libraries
numpy  
matplotlib  
scipy.integrate  
scipy.optimize  


## File Structure and Description

### Programming Tasks (PT)

PT_solve_ODE.py  
- Numerical solver for the equations of motion  
- Solves the ODE system for the third body  
- Generates trajectories from given initial conditions  

PT_Omega_contour.py  
- Produces contour plots of the effective potential Ω(x, y)  
- Visualizes allowed and forbidden regions of motion  


### Question 2

Q2.py  
- Implements the simplified model near P2  
- Uses approximate potential  
- Verifies numerical solution against analytic circular orbit  


### Question 3

Q3.py  
- Simulates trajectories for different initial velocities  
- Integrates motion over time and plots trajectories  

Q3_compare.py  
- Compares results with Question 1 outputs  
- Used for validation and consistency checking  


### Question 4

Q4_Omega_contour.py  
- Generates Ω contour plots for different values of μ  
- Identifies equilibrium (Lagrange) points  

Q4_stability.py  
- Studies stability of equilibrium points numerically  
- Simulates trajectories near equilibrium points  


### Question 5

Q5.py  
- Investigates stability of equilateral Lagrange points  
- Explores dependence on parameter μ  
- Used to estimate the critical stability value μ_c  


## How to Run

Run each file independently depending on the question:

python PT_solve_ODE.py  
python PT_Omega_contour.py  
python Q2.py  
python Q3.py  
python Q3_compare.py  
python Q4_Omega_contour.py  
python Q4_stability.py  
python Q5.py  


## Notes

- All parameters and initial conditions are defined within each script  
- No external input files are required  
- Plots are generated directly when running the scripts  
- Numerical accuracy should be checked by monitoring conserved quantities and adjusting solver settings  