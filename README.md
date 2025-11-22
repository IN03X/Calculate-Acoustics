# Calculate Acoustics
## Cal 1
### Equation Solving: Bubble Vibration
**Second-order** ordinary differential equation (**ODE**) in **one variable** with respect to time.
### Supports
Numerical schemes: RK1–RK4.

## Cal 2
### Equation Solving: Loranz System
**First-order** ordinary differential equation (**ODE**) in **three variables** with respect to time (The three variables are mutually independent).
### Supports
Numerical schemes: RK1–RK4.

## Cal 3
### Equation Solving: Steady-state 2D Acoustic Systems
**Second-order** **Steady-state** partial differential equation (**PDE**) in **two variables**.
### Supports
Equation models: Paraxial Approximation and Paraxial Approximation in Cylindrical Coordinates.\
<img width="179" height="62" alt="image" src="https://github.com/user-attachments/assets/7c0c5dec-203e-46ec-9f92-170cf740a450" />\
<img width="406" height="65" alt="image" src="https://github.com/user-attachments/assets/93f897e5-7da0-4c47-983a-cc697bd96db6" />\
Numerical schemes: Forward Euler, Dufort–Frankel, and Backward Euler.\
Acoustic sources: Piston and Focused source.\
Boundary: Reflective boundary.
### Tips
USE cal3.3.4_Acoustic_Wave_Solver.py\
IGNORE Evaluation because the analytical_solution is WRONG.

## Cal 4
### Boundary Setting: PML Condition
Example of a PML boundary, which can be used as an absorbing boundary.
