import os
import traceback
import numpy as np
from ansys.mapdl.core import launch_mapdl


def evaluate(L, W, T, F, run_id):
    """
    Runs a single FEA evaluation for given geometry/load parameters.
    Returns a dict with results (or NaNs + error info on failure).
    """
    run_dir = os.path.abspath(f"./runs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)  # launch_mapdl won't create this itself

    mapdl = None
    try:
        mapdl = launch_mapdl(
            run_location=run_dir,
            cleanup_on_exit=True,   # delete working files after this run finishes
            override=True,          # survive leftover lock files from prior crashes
        )

        # ---- Geometry ----
        mapdl.clear()
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

        # ---- Post-processing (NumPy arrays, not raw *GET) ----
        mapdl.post1()
        mapdl.set("LAST")

        max_stress = float(mapdl.post_processing.nodal_eqv_stress().max())
        max_defl = float(mapdl.post_processing.nodal_displacement("Y").min())

        return {
            "L": L, "W": W, "T": T, "F": F,
            "max_stress": max_stress,
            "max_defl": max_defl,
            "status": "ok",
        }

    except Exception as e:
        return {
            "L": L, "W": W, "T": T, "F": F,
            "max_stress": np.nan,
            "max_defl": np.nan,
            "status": f"failed: {e}",
            "traceback": traceback.format_exc(),
        }

    finally:
        # Guarantees MAPDL always shuts down, even on a mid-run crash —
        # this is what prevents orphaned processes/lock files.
        if mapdl is not None:
            try:
                mapdl.exit()
            except Exception:
                pass  # already dead or unreachable — nothing more to do


if __name__ == "__main__":
    result = evaluate(L=200, W=20, T=10, F=-500, run_id="test_001")
    print(result)