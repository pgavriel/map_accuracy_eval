import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Generate example sparse data
sparse_x = np.random.uniform(-5, 5, 50)
print(sparse_x)
sparse_y = np.random.uniform(-5, 5, 50)
sparse_z = np.sin(np.sqrt(sparse_x**2 + sparse_y**2))

# Create a regular grid for interpolation
grid_resolution = 250
x = np.linspace(-5, 5, grid_resolution)
y = np.linspace(-5, 5, grid_resolution)
X, Y = np.meshgrid(x, y)

# Interpolate sparse data onto the regular grid
Z = griddata((sparse_x, sparse_y), sparse_z, (X, Y), method='linear')


# Create a contour plot
plt.figure(figsize=(16, 9))
contour = plt.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
plt.colorbar(contour)

plt.title('Global Error Contour Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
