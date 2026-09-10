import os
from ansys.mapdl.core import launch_mapdl

os.makedirs("./debug_import", exist_ok=True)

CAD_DIR = os.path.abspath("../CAD")
CAD_NAME = "L mount minimum v1"

length = 0.02
width = 0.02
thickness = 0.002
height = 0.02
HOLE_RADIUS = 0.005
REMOTE_OFFSET_X = -0.030
FORCE_Y = -150.0

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

nodes = mapdl.mesh.nodes
print(f"X range: {nodes[:,0].min():.5f} to {nodes[:,0].max():.5f}")
print(f"Y range: {nodes[:,1].min():.5f} to {nodes[:,1].max():.5f}")
print(f"Z range: {nodes[:,2].min():.5f} to {nodes[:,2].max():.5f}")

# ---- Fixed support: cylindrical face of the hole ----
mapdl.allsel()
mapdl.csys(0)
mapdl.nsel("S", "LOC", "X", width / 2 - HOLE_RADIUS * 1.01, width / 2 + HOLE_RADIUS * 1.01)
mapdl.nsel("R", "LOC", "Y", -thickness - 0.001, 0.001)

n_fixed = mapdl.get_value("NODE", 0, "COUNT")
print(f"Nodes selected for fixed support: {n_fixed}")

mapdl.d("ALL", "ALL", 0)
mapdl.allsel()

# ---- Remote force: outer face of the upright leg ----
mapdl.nsel("S", "LOC", "X", 0)
mapdl.nsel("R", "LOC", "Y", 0, height)

n_face = mapdl.get_value("NODE", 0, "COUNT")
print(f"Nodes selected for remote-force face: {n_face}")

face_component = "UPRIGHT_FACE"
mapdl.cm(face_component, "NODE")
mapdl.allsel()

remote_x = REMOTE_OFFSET_X
remote_y = height / 2
remote_z = length / 2
mapdl.n(0, remote_x, remote_y, remote_z)
remote_node = int(mapdl.get_value("NODE", 0, "NUM", "MAX"))
print(f"Remote node created: {remote_node} at ({remote_x}, {remote_y}, {remote_z})")

# Select ONLY the master node, then ADD the slave face component to the
# same active selection. CERIG needs both in the selection, with "ALL"
# passed as the slave argument -- it does not accept component names.
mapdl.nsel("S", "NODE", "", remote_node)
mapdl.cmsel("A", face_component)

mapdl.cerig(remote_node, "ALL", "ALL")
mapdl.f(remote_node, "FY", FORCE_Y)

mapdl.allsel()
mapdl.eplot(savefig="mesh_with_bcs.png")
print("Saved mesh_with_bcs.png")

mapdl.exit()