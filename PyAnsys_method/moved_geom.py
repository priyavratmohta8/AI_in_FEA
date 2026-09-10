import os
from ansys.mapdl.core import launch_mapdl

os.makedirs("./debug_import", exist_ok=True)

CAD_DIR = os.path.abspath("../CAD")
CAD_NAME = "L mount minimum v1"

mapdl = launch_mapdl(run_location="./debug_import", override=True)
mapdl.clear()
mapdl.prep7()

mapdl.mp("EX", 1, 200e9)
mapdl.mp("NUXY", 1, 0.3)
mapdl.mp("DENS", 1, 7850)
mapdl.et(1, 187)

mapdl.parain(name=CAD_NAME, extension="x_t", path=CAD_DIR, entity="SOLIDS")

mapdl.esize(0.001)
mapdl.vmesh("ALL")

# ---- DIAGNOSTIC: find the actual coordinate extents of the imported geometry ----
nodes = mapdl.mesh.nodes  # numpy array, shape (n_nodes, 3)
print(f"X range: {nodes[:,0].min():.5f} to {nodes[:,0].max():.5f}")
print(f"Y range: {nodes[:,1].min():.5f} to {nodes[:,1].max():.5f}")
print(f"Z range: {nodes[:,2].min():.5f} to {nodes[:,2].max():.5f}")

mapdl.exit()