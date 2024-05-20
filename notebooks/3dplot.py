import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

siginreg = np.load("nonlinear_regions.npy")
reg = pd.read_csv("AAL_90regions.csv")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
linear = reg.iloc[siginreg==0]
mild = reg.iloc[np.logical_and(siginreg>0, siginreg<=3)]
very = reg.iloc[siginreg>3]

ax.scatter(linear.X, linear.Y, linear.Z, marker="o", color="gray")
ax.scatter(mild.X, mild.Y, mild.Z, marker="o", color="orange")
ax.scatter(very.X, very.Y, very.Z, marker="o", color="red")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()