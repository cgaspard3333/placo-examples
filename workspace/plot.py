import numpy as np
import pickle
from polytope import Polytope
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

import matplotlib.pyplot as plt

polytope_old = Polytope.load("workspace_old.pkl")
polytope_new = Polytope.load("workspace_new.pkl")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot polytope_old
M_old = np.hstack((polytope_old.A, -polytope_old.b.reshape(-1, 1)))
hd_old = HalfspaceIntersection(M_old, np.array([0.0, 0.0, 0.0]))
hull_old = ConvexHull(hd_old.intersections)

for i in hull_old.simplices:
    ax.plot(
        hd_old.intersections[i, 0],
        hd_old.intersections[i, 1],
        hd_old.intersections[i, 2],
        "g-",
    )

points_old = np.array([x * y for x, y in zip(polytope_old.A, polytope_old.b)])
ax.scatter(points_old[:, 0], points_old[:, 1], points_old[:, 2], color="b", label="Polytope Old")

# Plot polytope_new
M_new = np.hstack((polytope_new.A, -polytope_new.b.reshape(-1, 1)))
hd_new = HalfspaceIntersection(M_new, np.array([0.0, 0.0, 0.0]))
hull_new = ConvexHull(hd_new.intersections)

for i in hull_new.simplices:
    ax.plot(
        hd_new.intersections[i, 0],
        hd_new.intersections[i, 1],
        hd_new.intersections[i, 2],
        "r-",
    )

points_new = np.array([x * y for x, y in zip(polytope_new.A, polytope_new.b)])
ax.scatter(points_new[:, 0], points_new[:, 1], points_new[:, 2], color="r", label="Polytope New")

ax.legend()
plt.show()

