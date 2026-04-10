import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def local_orbit(t, z, mu):
    x, y, u, v = z

    dx = x - mu
    dy = y
    r = np.sqrt(dx**2 + dy**2)

    dOmega_dx = dx / (2 * r**3)
    dOmega_dy = dy / (2 * r**3)

    dxdt = u
    dydt = v
    dudt = 2 * v - dOmega_dx
    dvdt = -2 * u - dOmega_dy

    return [dxdt, dydt, dudt, dvdt]

mu = 0.5
a = 0.05

omega = -1 + np.sqrt(1 + 1/(2*a**3))

z0 = [mu + a, 0.0, 0.0, a * omega]

t_span = (0.0, 5.0)
t_eval = np.linspace(t_span[0], t_span[1], 3000)

sol = solve_ivp(
    local_orbit,
    t_span,
    z0,
    args=(mu,),
    method="RK45",
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

t = sol.t
x_num = sol.y[0]
y_num = sol.y[1]

x_exact = mu + a * np.cos(omega * t)
y_exact = a * np.sin(omega * t)

error = np.sqrt((x_num - x_exact)**2 + (y_num - y_exact)**2)
print("success:", sol.success)
print("Maximum position error =", np.max(error))
