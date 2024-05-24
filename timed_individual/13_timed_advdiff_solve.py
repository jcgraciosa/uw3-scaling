# %%
import os
os.environ["UW_TIMING_ENABLE"] = "1"

#import nest_asyncio
#nest_asyncio.apply()

import numpy as np
from pathlib import Path

# %%
# Set some things
import sympy
from sympy import Piecewise

import underworld3 as uw
from underworld3.systems import Stokes # should we also include other systems?
from underworld3 import function
from underworld3 import timing

# %%
n_els           = int(os.getenv("UW_RESOLUTION",16))
dim             = int(os.getenv("UW_DIM",2)) # FIXME: add option for running 3D model
scaling_type    = int(os.getenv("SCALING_TYPE",1))
uw_name         = str(os.getenv("UW_NAME", ""))
uw_model        = str(os.getenv("UW_MODEL", ""))

tol     = float(os.getenv("UW_SOL_TOLERANCE",1.e-5))
max_its = int(os.getenv("UW_MAX_ITS",-1))

# %%
debug       = True
do_timing   = True
refinement  = None

# BC setting
mode_use = 1 # 0 for the old version, 1 for the one with sympy.oo

eta_0 = 1.0
f_0 = 1.0

x_c = 0.5

# %%
if do_timing:
    timing.reset()
    timing.start()

mesh = uw.meshing.UnstructuredSimplexBox(regular=False,
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / n_els,
    qdegree=3, refinement = refinement
)

#mesh.dm.view()

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2, varsymbol = r"V")
t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree = 3, continuous = True, varsymbol = r"{T}")

# # %%
# Advection-Diffusion
k = 1.0
h = 0.0

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=t_soln,
    V_fn=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5

adv_diff.add_dirichlet_bc(0.0, "Bottom")
adv_diff.add_dirichlet_bc(1.0, "Top")

adv_diff.tolerance = tol
adv_diff.petsc_options["snes_max_it"] = max_its
adv_diff.petsc_options["ksp_max_it"] = max_its

# %%
x,y = mesh.X

# set to a fixed velocity field if using max_its
if max_its >= 0:
    # define velocity sympy function
    v_fn = sympy.Matrix(
    [
        Piecewise(
            (-1, ((x >= 0.5) & (y >= 0.5))),
            (0, ((x <= 0.5) & (y >= 0.5))),
            (1, ((x <= 0.5) & (y <= 0.5))),
            (0, ((x >= 0.5) & (y <= 0.5))),
            (0.0, True),
        ),
        Piecewise(
            (0, ((x >= 0.5) & (y >= 0.5))),
            (-1, ((x <= 0.5) & (y >= 0.5))),
            (0, ((x <= 0.5) & (y <= 0.5))),
            (1, ((x >= 0.5) & (y <= 0.5))),
            (0.0, True),
        ),
    ]
)
    with mesh.access(v_soln):
        v_soln.data[...] = uw.function.evaluate(v_fn, v_soln.coords).reshape(-1, 2)

# initialize temperature field
with mesh.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(y, t_soln.coords).reshape(-1, 1)

# # %%
# # Advection-Diffusion
if uw.mpi.rank == 0 and debug:
    print("2. Attempting adv_diff solve ... ")
adv_diff.solve(timestep=0.001, zero_init_guess=True)
if uw.mpi.rank == 0 and debug:
    print("2. Solved adv_diff ... ")

if uw.mpi.rank == 0:
    with mesh.access():
        print(v_soln.data.min())
        print(v_soln.data.max())
        print(t_soln.data.min())
        print(t_soln.data.max())

# %%
# check the mesh if in a notebook / serial

# %%
### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
#outputPath = f"../output/timing-tests/"
if scaling_type == 1:
    outputPath = f"../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-tests-2D-weak-{uw_name}-{uw_model}/"
elif scaling_type == 2:
    outputPath = f"../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-tests-2D-strong-{uw_name}-{uw_model}/"

### write test
if uw.mpi.rank==0:
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

# %%
if do_timing:
    timing.stop()

uw.mpi.barrier()
if uw.mpi.rank == 0:
    for fname_end in ("*.h5", "*.xdmf", "*.xmf", "*.pbin"):
        for filename in Path(outputPath).glob(fname_end):
            os.remove(filename)

# %%

if do_timing:
    module_timing_data_orig = uw.timing.get_data(group_by="routine")

    # write out data
    #filename = "Res_{}_Nproc_{}_JobID_{}".format(res,uw.mpi.size,jobid)
    filename = "Res_{}_Nproc_{}".format(n_els,uw.mpi.size)
    import json
    if module_timing_data_orig:
        module_timing_data = {}
        for key,val in module_timing_data_orig.items():
            module_timing_data[key[0]] = val
        #module_timing_data["Other_timing"] = other_timing
        #module_timing_data["Other_data"]   = { "res":res, "nproc":uw.mpi.size, "vrms":vrms, "prms":prms }
        with open(f"{outputPath}/{filename}.json", 'w') as fp:
            json.dump(module_timing_data, fp,sort_keys=True, indent=4)

    uw.timing.print_table(group_by="routine", output_file=f"{outputPath}/{filename}.txt", display_fraction = 1.00)

    if uw.mpi.rank == 0 and debug:
        print(module_timing_data)


