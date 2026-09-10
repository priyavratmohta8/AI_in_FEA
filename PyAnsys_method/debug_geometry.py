import os
from ansys.mapdl.core import launch_mapdl

os.makedirs("./debug_geom", exist_ok=True)

mapdl = launch_mapdl(run_location="./debug_geom", override=True)
mapdl.clear()
mapdl.prep7()

length = 0.04
width = 0.035
thickness = 0.003
height = 0.04
FILLET_RADIUS = 0.003
HOLE_RADIUS = 0.005

# ---- STEP 1: L-profile sketch, XY plane ----
k1 = mapdl.k(1, 0, 0)
k2 = mapdl.k(2, width, 0)
k3 = mapdl.k(3, width, thickness)
k4 = mapdl.k(4, thickness, thickness)
k5 = mapdl.k(5, thickness, thickness + height)
k6 = mapdl.k(6, 0, thickness + height)

l1 = mapdl.l(k1, k2)
l2 = mapdl.l(k2, k3)
l3 = mapdl.l(k3, k4)
l4 = mapdl.l(k4, k5)
l5 = mapdl.l(k5, k6)
l6 = mapdl.l(k6, k1)

# ---- STEP 2: fillet the internal corner ----
mapdl.lfillt(l3, l4, FILLET_RADIUS)
mapdl.allsel()
mapdl.al("ALL")

# ---- STEP 3: extrude into a solid (BEFORE touching the working plane) ----
mapdl.vext("ALL", dz=length)

mapdl.vplot(show_lines=True, savefig="geom_check.png")
print("Saved geom_check.png -- L-shape + fillet, no hole yet.")

# ---- STEP 4: hole, cut AFTER the solid already exists ----
mapdl.wpcsys(-1, 0)
mapdl.wprota(90, 0, 0)
mapdl.wpoffs(width / 2, length / 2, 0)

eps = thickness * 0.1
mapdl.cyl4(0, 0, HOLE_RADIUS, depth=thickness + 2 * eps)

mapdl.wpcsys(-1, 0)  # reset WP back to global immediately after the cut geometry is made

mapdl.vplot(show_lines=True, savefig="geom_with_cyl.png")
print("Saved geom_with_cyl.png -- check the cylinder position before subtracting.")

mapdl.exit()