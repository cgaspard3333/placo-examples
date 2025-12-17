import numpy as np
from polytope import Polytope
from scipy.spatial import ConvexHull, HalfspaceIntersection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

# --- Load polytope (new only)
polytope_new = Polytope.load("workspace_new.pkl")
polytope_new2 = Polytope.load("workspace_new2.pkl")

# --- Compute intersections (vertices) from half-spaces A x <= b
M_new = np.hstack((polytope_new.A, -polytope_new.b.reshape(-1, 1)))
hd_new = HalfspaceIntersection(M_new, np.array([0.0, 0.0, 0.0]))
verts = hd_new.intersections

# --- Convex hull faces
hull = ConvexHull(verts)
faces = [verts[s] for s in hull.simplices]

# --- Face colors (viridis by centroid radius)
centroids = np.array([f.mean(axis=0) for f in faces])
r = np.linalg.norm(centroids, axis=1)
r_norm = (r - r.min()) / (r.max() - r.min()) if r.ptp() != 0 else np.zeros_like(r)
face_colors = cm.get_cmap("viridis")(r_norm)

# --- Point cloud from your original script (A_i * b_i, row-wise)
points_new = np.array([rowA * bi for rowA, bi in zip(polytope_new2.A, polytope_new2.b)])

# --- Common axis limits across all views
all_pts = np.vstack([verts, points_new])
xlim = (all_pts[:, 0].min(), all_pts[:, 0].max())
ylim = (all_pts[:, 1].min(), all_pts[:, 1].max())
zlim = (all_pts[:, 2].min(), all_pts[:, 2].max())

def add_polytope(ax, show_label=False):
    poly = Poly3DCollection(
        faces,
        facecolors=face_colors,
        edgecolors="black",
        linewidths=1.2,
        alpha=0.6
    )
    ax.add_collection3d(poly)

    # joyful points
    scatter_label = "points_new" if show_label else None
    ax.scatter(
        points_new[:, 0], points_new[:, 1], points_new[:, 2],
        s=4, c="#ff7f0e", alpha=0.7, depthshade=True, label=scatter_label
    )

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

def set_axes_visibility(ax, show_x, show_y, show_z):
    # Labels
    ax.set_xlabel(r"$\Delta x$ (m)" if show_x else "")
    ax.set_ylabel(r"$\Delta y$ (m)" if show_y else "")
    ax.set_zlabel(r"$\Delta \theta$ (rad)" if show_z else "")
    # Ticks: only modify when hiding
    if not show_x:
        ax.set_xticks([])
    if not show_y:
        ax.set_yticks([])
    if not show_z:
        ax.set_zticks([])

# --- Figure with 3 synchronized views
fig = plt.figure(figsize=(15, 5))

# 1) Top view (Δx–Δy plane) — show x & y only
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
add_polytope(ax1, show_label=False)
ax1.view_init(elev=90, azim=-90)  # look straight down +z
set_axes_visibility(ax1, show_x=True, show_y=True, show_z=False)
ax1.set_title(r"Top view ($\Delta x$–$\Delta y$ plane)")

# 2) Side view (Δy–Δθ plane) — show y & Δθ only
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
add_polytope(ax2, show_label=False)
ax2.view_init(elev=0, azim=0)     # look along +x so y–z plane is front-facing
set_axes_visibility(ax2, show_x=False, show_y=True, show_z=True)
ax2.set_title(r"Side view ($\Delta y$–$\Delta \theta$ plane)")

# 3) Isometric view — show all + legend
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
add_polytope(ax3, show_label=True)
ax3.view_init(elev=25, azim=-60)
set_axes_visibility(ax3, show_x=True, show_y=True, show_z=True)
ax3.set_title(r"3D view in $(\Delta x,\ \Delta y,\ \Delta \theta)$")
ax3.legend(loc="best")

# Remove legend added above
leg = ax3.get_legend()
if leg is not None:
    leg.remove()

plt.tight_layout()
plt.show()
