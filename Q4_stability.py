import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

mu = 0.001
target = "L1"     
eps = 1e-3     
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 3000)

x1 = -1 + mu
x2 = mu

def distances(x, y):
    r1 = np.sqrt((x - x1)**2 + y**2)
    r2 = np.sqrt((x - x2)**2 + y**2)
    return r1, r2

def dU_dx(x, y):
    r1, r2 = distances(x, y)
    return x - mu*(x - x1)/r1**3 - (1 - mu)*(x - x2)/r2**3

def dU_dy(x, y):
    r1, r2 = distances(x, y)
    return y - mu*y/r1**3 - (1 - mu)*y/r2**3

def cr3bp_equations(t, state):
    x, y, vx, vy = state
    ax = 2*vy + dU_dx(x, y)
    ay = -2*vx + dU_dy(x, y)
    return [vx, vy, ax, ay]

def collinear_eq(x):
    return dU_dx(x, 0.0)

L3 = (fsolve(collinear_eq, -1.2)[0], 0.0)
L4 = (fsolve(collinear_eq,  0.0)[0], 0.0)
L5 = (fsolve(collinear_eq,  1.2)[0], 0.0)
L1 = (mu - 0.5,  np.sqrt(3)/2)
L2 = (mu - 0.5, -np.sqrt(3)/2)

equilibria = {
    "L1": L1,
    "L2": L2,
    "L3": L3,
    "L4": L4,
    "L5": L5
}

xeq, yeq = equilibria[target]

y0 = [xeq+eps, yeq, 0.0, 0.0]

sol = solve_ivp(
    cr3bp_equations,
    t_span,
    y0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-12
)

t = sol.t
x = sol.y[0]
y = sol.y[1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(x, y, label=f"Trajectory near {target}")
axes[0].scatter([x1, x2], [0, 0], s=80, label="large bodies")
axes[0].scatter([xeq], [yeq], s=80, marker='x', label=target)

axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("(x, y) trajectory")
axes[0].axis("equal")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t, x)
axes[1].axhline(xeq, linestyle='--', label='equilibrium x')

axes[1].set_title("x(t)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("x")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(t, y)
axes[2].axhline(yeq, linestyle='--', label='equilibrium y')

axes[2].set_title("y(t)")
axes[2].set_xlabel("t")
axes[2].set_ylabel("y")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()