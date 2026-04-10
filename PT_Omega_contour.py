import numpy as np
import matplotlib.pyplot as plt

mu = 0.3

def Omega(x, y, mu):
    r1 = np.sqrt((x + 1 - mu)**2 + y**2)
    r2 = np.sqrt((x - mu)**2 + y**2)
    
    return (
        -0.5 * mu * r1**2
        -0.5 * (1 - mu) * r2**2
        - mu / r1
        - (1 - mu) / r2
    )

xg = np.linspace(-3.0, 3.0, 600)
yg = np.linspace(-3.0, 3.0, 600)
X, Y = np.meshgrid(xg, yg)

Z = Omega(X, Y, mu)

Z[np.isinf(Z)] = np.nan

levels = np.linspace(-5, 1, 30)

plt.figure(figsize=(7,7))

contours = plt.contour(X, Y, Z, levels=levels)
plt.clabel(contours, inline=True, fontsize=8)

plt.scatter(-1 + mu, 0, c='red', s=80, label='body 1')
plt.scatter(mu, 0, c='blue', s=80, label='body 2')

plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.show()