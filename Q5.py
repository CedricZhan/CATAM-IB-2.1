import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

eps = 1e-3
t_span = (0, 80)
npts = 5000
t_eval = np.linspace(t_span[0], t_span[1], npts)

mu_values = [0.008, 0.022, 0.044, 0.08, 0.5]

def make_system(mu):
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

    return x1, x2, cr3bp_equations

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, mu in enumerate(mu_values):
    x1, x2, cr3bp_equations = make_system(mu)

    L1 = (mu - 0.5,  np.sqrt(3)/2)

    xeq, yeq = L1
    y0 = [xeq + eps, yeq, 0.0, 0.0]

    sol = solve_ivp(
        cr3bp_equations,
        t_span,
        y0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )

    x = sol.y[0]
    y = sol.y[1]

    ax = axes[i]
    ax.plot(x, y, lw=1, label="trajectory")
    ax.scatter([x1, x2], [0, 0], s=50, label="large bodies")
    ax.scatter([xeq], [yeq], s=60, marker='x', label="L1")

    ax.set_title(f"$\\mu={mu}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8)

if len(mu_values) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()