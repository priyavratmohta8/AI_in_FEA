import os
import numpy as np
from ansys.mapdl.core import launch_mapdl

# ---- Launch ----
run_dir = "./mapdl_hello_world"
os.makedirs(run_dir, exist_ok=True)

mapdl = launch_mapdl(run_location=run_dir, cleanup_on_exit=False, override=True)
mapdl.clear()

# ---- Parameters ----
L = 200   # length, mm
W = 20    # width, mm
T = 10    # thickness, mm
F = -500  # tip load, N (downward)

# ---- Geometry ----
mapdl.prep7()
mapdl.block(0, L, 0, W, 0, T)

# ---- Material ----
mapdl.mp("EX", 1, 210000)
mapdl.mp("PRXY", 1, 0.3)

# ---- Element type + meshing ----
mapdl.et(1, "SOLID186")
mapdl.esize(5)
mapdl.vmesh("ALL")

# ---- Boundary conditions ----
mapdl.solution()
mapdl.antype("STATIC")

mapdl.nsel("S", "LOC", "X", 0)
mapdl.d("ALL", "ALL", 0)
mapdl.allsel()

mapdl.nsel("S", "LOC", "X", L)
n_nodes = mapdl.get_value("NODE", 0, "COUNT")
if n_nodes == 0:
    raise RuntimeError("No nodes found at load face — check geometry/mesh")
mapdl.f("ALL", "FY", F / n_nodes)
mapdl.allsel()

# ---- Solve ----
mapdl.solve()
mapdl.finish()

# ---- Post-processing ----
mapdl.post1()
mapdl.set("LAST")

max_stress = mapdl.post_processing.nodal_eqv_stress().max()
max_defl   = mapdl.post_processing.nodal_displacement("Y").min()

print(f"Max von Mises stress: {max_stress:.2f} MPa")
print(f"Max tip deflection:   {max_defl:.4f} mm")

# ---- Save results ----
result = mapdl.result
result.plot_nodal_displacement(0, show_edges=True, screenshot="deflection.png")
result.plot_principal_nodal_stress(0, "SEQV", show_edges=True, screenshot="stress.png")

np.savetxt("summary.csv",
           [[L, W, T, F, max_stress, max_defl]],
           delimiter=",",
           header="L,W,T,F,max_stress,max_defl",
           comments="")

mapdl.exit()