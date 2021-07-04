# code from https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
"""
Used for generating coloured interpolated graph over the test dataset loss
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

df = pd.read_csv("../regression/testData/loss.csv")

points = df[["yreal", "yimag"]].to_numpy()
z = df["loss"].to_numpy()

r = 3.2

grid_x, grid_y = np.mgrid[-r:r:1000j, -r:r:1000j]
Z = griddata(points, z, (grid_x, grid_y), method='cubic', fill_value=100)


plt.imshow(Z, extent=(-r,r,-r,r), vmin=0, vmax=7, origin='lower', cmap="jet")
plt.title("Interpolated prediction error")
plt.xlabel('Real', fontsize=10)
plt.ylabel('Imaginary', fontsize=10)
clb=plt.colorbar()
clb.ax.tick_params(labelsize=8) 
clb.ax.set_title('Loss',fontsize=10)
plt.show()