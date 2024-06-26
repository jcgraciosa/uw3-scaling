# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: uw3-dev
#     language: python
#     name: python3
# ---

# %%
import os
os.environ["UW_TIMING_ENABLE"] = "1"
import underworld3 as uw
#uw.tools.parse_cmd_line_options()
import numpy as np
import time
import math
from petsc4py import PETSc
from underworld3.systems import Stokes
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank

import sympy
from sympy import Piecewise, ceiling, Abs

# %%
order = int(os.getenv("UW_ORDER", "1" ))
res   = int(os.getenv("UW_RESOLUTION", 16)) # originally 32
dim   = int(os.getenv("UW_DIM", 3 ))

# %%
otol  = float(os.getenv("UW_SOL_TOLERANCE", 1.e-6))
max_its  = int(os.getenv("UW_MAX_ITS", -1))

# %%
jobid = str(os.getenv("PBS_JOBID",os.getenv("SLURM_JOB_ID","0000000")))

# %%
picklename = str(os.getenv("PICKLENAME","None"))

outputPath = "../../../../../scratch/el06/jg0883/uw3-scaling/output/timing-tests-OLD-3D/"

### write test
if uw.mpi.rank==0:
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

# %%
# # Find all available solutions.
# # Use ordered dict to preserve alphabetical ordering
# import collections
# solns_avail = collections.OrderedDict()
# for _soln in dir(fn.analytic):
#     if _soln[0] == "_": continue  # if private member, ignore
#     # get soln class
#     soln = getattr(fn.analytic,_soln)
#     # check if actually soln
#     if issubclass(soln, fn.analytic._SolBase):
#         solns_avail[_soln] = soln
# soln = solns_avail[soln_name]()


# %%
time_post_import   = time.time()
time_launch_mpi    = float(os.getenv("TIME_LAUNCH_MPI"   ,time_post_import))/1000.
time_launch_python = float(os.getenv("TIME_LAUNCH_PYTHON",time_post_import))/1000.

# %%
uw.timing.start()
stime = time.time()

# %%
other_timing = {}
other_timing["Python_Import_Time"] = time_post_import - time_launch_python
other_timing[   "MPI_Launch_Time"] = time_launch_python - time_launch_mpi


# %%
options = PETSc.Options()

# %%
if max_its < 0:
    options["ksp_rtol"] =  otol
else:
    options["ksp_rtol"] =  1e-99
    options["ksp_max_it"] = max_its

# %%
boxLength      = 0.9142
ppcell         = 1
amplitude  = 0.02
offset     = 0.2
viscosityRatio = 1.0

# %%
# options["help"] = None
# options["pc_type"]  = "svd"
# options["ksp_atol"] =  1.0e-6
options["ksp_monitor"] = None
# options["snes_type"]  = "fas"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
# options["snes_view"]=None
# options["snes_test_jacobian"] = None
# options["snes_rtol"] = 1.0e-2  # set this low to force single SNES it.
options["snes_max_it"] = 1
# options["pc_type"] = "fieldsplit"
# options["pc_fieldsplit_type"] = "schur"
# options["pc_fieldsplit_schur_factorization_type"] = "full"
# # options["fieldsplit_pressure_ksp_rtol"] = 1e-6
# options["fieldsplit_velocity_pc_type"] = "lu"
# options["fieldsplit_pressure_pc_type"] = "jacobi"
# options["fieldsplit_velocity_ksp_type"] = "gmres"
# sys = PETSc.Sys()
# sys.pushErrorHandler("traceback")

# %%
mesh   = uw.meshing.UnstructuredSimplexBox(regular = False, cellSize=1/res,minCoords=(0.,)*dim,maxCoords=(1.,)*dim)
if rank==0: print(f"runtime {time.time()-stime}s, mesh init done.")
stokes = Stokes(mesh)

v = stokes.Unknowns.u
p = stokes.Unknowns.p
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

if rank==0: print(f"runtime {time.time()-stime}s, stokes init done.")
swarm  = uw.swarm.Swarm(mesh)
# Add variable for material
#matSwarmVar = swarm.add_variable(name="matSwarmVar",  size=1,   dtype=PETSc.IntType)

matSwarmVar = uw.swarm.IndexSwarmVariable("matSwarmVar", swarm, indices=2, proxy_continuous=False, proxy_degree=1)
velSwarmVar = swarm.add_variable(name="velSwarmVar",  size=dim, dtype=PETSc.ScalarType)
# Note that `ppcell` specifies particles per cell per dim.
swarm.populate(fill_param = 4)

# %%
if rank==0: print(f"runtime {time.time()-stime}s, swarm init and populate done.")

# %%
import numpy as np
with swarm.access():
    vel_on_particles = uw.function.evaluate(stokes.u.fn,swarm.particle_coordinates.data[0:3])
np.random.seed(0)
with swarm.access(swarm.particle_coordinates):
    factor = 0.5*boxLength/res/ppcell
    swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)

# %%
if rank==0: print(f"runtime {time.time()-stime}s, particle random done.")
# define these for convenience.
denseIndex = 0
lightIndex = 1

# %%
# material perturbation from van Keken et al. 1997
wavelength = 2.0*boxLength
k = 2. * np.pi / wavelength

# %%
# init material variable
with swarm.access(matSwarmVar):
    perturbation = offset + amplitude*np.cos( k*swarm.particle_coordinates.data[:,0] )
    matSwarmVar.data[:,0] = np.where( perturbation>swarm.particle_coordinates.data[:,1], lightIndex, denseIndex )

# %%
# density = Piecewise( ( 0., Abs(matSwarmVar.sym - lightIndex)<0.5 ),
#                         ( 1., Abs(matSwarmVar.sym - denseIndex)<0.5 ),
#                         ( 0.,                                True ) )

density = 0*matSwarmVar.sym[0] + 1*matSwarmVar.sym[1]
density

# %%
stokes.bodyforce = -density*mesh.N.j
stokes.bodyforce

# %%
# stokes.viscosity = Piecewise( ( viscosityRatio, Abs(matSwarmVar.fn - lightIndex)<0.5 ),
#                                 (             1., Abs(matSwarmVar.fn - denseIndex)<0.5 ),
#                                 (             1.,                                True ) )
stokes.viscosity = viscosityRatio * matSwarmVar.sym[0] + 1 * matSwarmVar.sym[1]
stokes.viscosity

# %%
bnds = mesh.boundaries
mesh.boundaries.Top

# %%
# note with petsc we always need to provide a vector of correct cardinality.
bnds = mesh.boundaries
stokes.add_dirichlet_bc((0.0, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")

# stokes.add_dirichlet_bc( (0.,0.), [bnds.TOP,  bnds.BOTTOM], (0,1) )  # top/bottom: function, boundaries, components
# stokes.add_dirichlet_bc( (0.,0.), [bnds.LEFT, bnds.RIGHT ], 0  )  # left/right: function, boundaries, components

# %%
volume_int = uw.maths.Integral( mesh, 1. )
volume = volume_int.evaluate()
v_dot_v_int = uw.maths.Integral(mesh, stokes.u.fn.dot(stokes.u.fn))

# %%
stokes.is_setup


# %%
if rank==0: print(f"runtime {time.time()-stime}s, mat var configured.")
# Solve time
stokes.solve(zero_init_guess = True)

# %%
if rank==0: print(f"runtime {time.time()-stime}s, solve done.")
# Create a fixed solid body like rotation to stress
# particle advection. Note that this only creates a
# 2d flow, and might be important to do a 3d flow to
# really test parallel topology.
vel = np.zeros(dim)
with mesh.access(stokes.u):
    for index,coord in enumerate(stokes.u.coords):
        tcoord = coord - (0.5,)*dim
        # force to zero at boundaries
        fact = (1.-4.*tcoord[0]**2)*(1-4*tcoord[1]**2)
        vel[0] = -fact*tcoord[1]
        vel[1] =  fact*tcoord[0]
        stokes.u.data[index] = vel

# %%
# import plot
# figs = plot.Plot(rulers=True)
# fig.edges(mesh)
# with swarm.access(),mesh.access():
#     figs.swarm_points(swarm, matSwarmVar.data, pointsize=4, colourmap="blue green", colourbar=False, title=time)
#     figs.vector_arrows(mesh, stokes.u.data)
#     fig.nodes(mesh,matMeshVar.data,colourmap="blue green", pointsize=6, pointtype=4)
# figs.image("velfield")

# %%
vrms = math.sqrt(v_dot_v_int.evaluate()/volume)

# %%
dt = stokes.estimate_dt()

# %%
# #Note that the evaluate method is way too slow at the moment. We won't use it.
with swarm.access(velSwarmVar):
    velSwarmVar.data[...] = uw.function.evaluate(stokes.u.fn,swarm.particle_coordinates.data[...])

swarm.advection(stokes.u.fn, 1 * stokes.estimate_dt())


# %%
# # Instead, as we have a closed form velocity, use that directly.
# with swarm.access(velSwarmVar):
#     for index,coord in enumerate(swarm.particle_coordinates.data):
#         tcoord = coord - (0.5,)*dim
#         # force to zero at boundaries
#         fact = (1.-4.*tcoord[0]**2)*(1-4*tcoord[1]**2)
#         vel[0] = -fact*tcoord[1]
#         vel[1] =  fact*tcoord[0]
#         velSwarmVar.data[index] = vel
# if rank==0: print(f"runtime {time.time()-stime}s, particle velocity done.")

# %%
# with swarm.access(swarm.particle_coordinates):
#     swarm.particle_coordinates.data[:]+=dt*velSwarmVar.data[:]

# %%
if rank==0: print(f"runtime {time.time()-stime}s, particle advect done.")

# %%
if MPI.COMM_WORLD.rank==0: print(f"VRMS = {vrms}")


# %%
uw.timing.stop()
module_timing_data_orig = uw.timing.get_data(group_by="routine")

# %%
# write out data
filename = "Res_{}_Nproc_{}_JobID_{}".format(res,uw.mpi.size,jobid)
import json
if module_timing_data_orig:
    module_timing_data = {}
    for key,val in module_timing_data_orig.items():
        module_timing_data[key[0]] = val
    other_timing["Total_Runtime"] = uw.timing._endtime-uw.timing._starttime
    module_timing_data["Other_timing"] = other_timing
    module_timing_data["Other_data"]   = { "res":res, "nproc":uw.mpi.size, "vrms":vrms,  }
    with open(f"{outputPath}/{filename}.json", 'w') as fp:
        json.dump(module_timing_data, fp,sort_keys=True, indent=4)

# %%
uw.timing.print_table(group_by="routine", output_file=filename+".txt", display_fraction=0.99)
