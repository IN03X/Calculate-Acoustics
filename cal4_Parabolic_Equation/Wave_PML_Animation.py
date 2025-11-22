import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Grid and physical parameters
# -----------------------------
L = 1.0               # domain length (m)
nx = 400              # number of grid points
dx = L / nx
x = np.linspace(0, L, nx)
c = 1.0               # wave speed (m/s)

# Time stepping (CFL)
CFL = 0.5
dt = CFL * dx / c
nt = 1500             # number of time steps to simulate

# -----------------------------
# PML setup: only on right side
# -----------------------------
L1 = 0.5 * L          # PML starts here
L2 = 0.8 * L                # domain end
pml_start = int(L1 / dx)
pml_end = int(L2 / dx)

# Define sigma(x) with two-stage damping: quadratic then cubic
sigma = np.zeros_like(x)
sigma_max = 100.0
for i in range(pml_start, nx):
    xi = (x[i] - L1) / (L2 - L1)  # 0 to 1 within PML
    if xi < 0.5:
        sigma[i] = sigma_max * (2*xi)**2      # quadratic in first half
    else:
        sigma[i] = sigma_max * (2*xi - 1)**3  # cubic in second half

# -----------------------------
# Initialize fields
# -----------------------------
p = np.zeros(nx)
u = np.zeros(nx)

# Sinusoidal source at x=0
f = 20.0  # Hz
omega = 2 * np.pi * f

# Second derivative function
def laplacian(p):
    d2 = np.zeros_like(p)
    d2[1:-1] = (p[2:] - 2*p[1:-1] + p[:-2]) / dx**2
    d2[0] = (p[1] - p[0]) / dx**2  # simple Neumann
    d2[-1] = (p[-2] - p[-1]) / dx**2
    return d2

# -----------------------------
# Animation setup
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, p, lw=2)
ax.axvline(L1, color='r', linestyle='--', label='PML start')
ax.axvline(L2, color='r', linestyle='--', label='PML end')
ax.legend()
ax.set_xlim(0, L)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('x (m)')
ax.set_ylabel('Pressure p')
ax.set_title('1D Continuous Sine Source with Right PML')
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

# -----------------------------
# Time stepping + animation
# -----------------------------
def update(n):
    global p, u
    d2p = laplacian(p)
    u = u + dt * (c**2 * d2p - sigma * u)
    p_new = p + dt * u

    # Boundary condition: continuous sine at x=0
    p_new[0] = np.sin(omega * n * dt)
    # Right boundary: hold previous value (absorbing-like)
    p_new[-1] = p[-1]

    p[:] = p_new

    line.set_ydata(p)
    time_text.set_text(f't = {n*dt:.3f} s')
    return line, time_text

ani = FuncAnimation(fig, update, frames=nt, interval=10, blit=True)
plt.show()

# Plot sigma profile
fig2, ax2 = plt.subplots(figsize=(8, 2))
ax2.plot(x, sigma)
ax2.axvline(L1, color='r', linestyle='--', label='PML start')
ax2.axvline(L2, color='r', linestyle='--', label='PML end')
ax2.legend()
ax2.set_xlabel('x (m)')
ax2.set_ylabel('sigma (1/s)')
ax2.set_title('PML damping profile: quadratic then cubic')
plt.tight_layout()
plt.show()