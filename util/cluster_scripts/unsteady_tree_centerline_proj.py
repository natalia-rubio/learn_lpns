"""
Legacy one-off 3D→centerline projection for a single CCO tree on Sherlock.

Projects only branch_id==0 (main trunk) points and subsamples timesteps (every 20th frame).
Hardcoded paths under /scratch/users/nrubio/synthetic_junctions/CCO/.

Usage: python unsteady_tree_centerline_proj.py <tree_name> <num_procs>
"""

import os
import sys
import vtk
from tqdm import tqdm
from util.get_bc_integrals import get_res_names
from util.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration, collect_arrays
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v
import pickle


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def slice_vessel(inp_3d, origin, normal):
    """
    Slice 3d geometry at certain plane
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # cut 3d geometry
    cut_3d = cut_plane(inp_3d, origin, normal)
    #write_geo(f'slice_{origin[0]}.vtp', cut_3d.GetOutput())

    # extract region closest to centerline
    con = connectivity(cut_3d, origin)
    #write_geo(f'con_{origin[0]}.vtp', con.GetOutput())
    return con

def get_integral(inp_3d, origin, normal):
    """
    Slice simulation at certain plane and integrate
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # slice vessel at given location
    inp = slice_vessel(inp_3d, origin, normal)

    # recursively add calculators for normal velocities

    for v in get_res_names(inp_3d, 'Velocity'):
        #fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        fun = (
            "dot(iHat*"
            + repr(float(normal[0]))
            + "+jHat*"
            + repr(float(normal[1]))
            + "+kHat*"
            + repr(float(normal[2]))
            + ","
            + v
            + ")"
        )
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)

tree_name = sys.argv[1]  # e.g., "tree_20_flow_unsteady"
num_procs = sys.argv[2]  # e.g., "96-procs"

input_file_names = os.listdir(f"/scratch/users/nrubio/synthetic_junctions/CCO/{tree_name}/{tree_name}_flow_unsteady/{num_procs}-procs")
input_file_names.sort()
times = [name.split('_')[-1].split('.')[0] for name in input_file_names if name.endswith('.vtu')]
res_names_1d = [f"pressure_{time}" for time in times] + [f"velocity_{time}" for time in times]

fpath_out = f"/scratch/users/nrubio/synthetic_junctions/CCO/{tree_name}/{tree_name}_flow_unsteady/unsteady_soln.vtp"

fpath_1d = f"/scratch/users/nrubio/synthetic_junctions/CCO/{tree_name}/{tree_name}_flow_unsteady/centerlines/centerlines.vtp"  # Assuming the first file is the 1D centerline
fpath_3d = f"/scratch/users/nrubio/synthetic_junctions/CCO/{tree_name}/{tree_name}_flow_unsteady/{num_procs}-procs/{tree_name}_unsteady_result_{times[0]}.vtu"
reader_1d = read_geo(fpath_1d).GetOutput()
reader_3d = read_geo(fpath_3d).GetOutput()# get all result array names
res_names_3d = get_res_names(reader_3d, ['Pressure', 'Velocity'])# get point and normals from centerline
points = v2n(reader_1d.GetPoints().GetData())
normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))# initialize output
branch_id = v2n(reader_1d.GetPointData().GetArray('BranchId'))
for name in res_names_1d + ['area']:
    array = vtk.vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
    array.Fill(float('NaN'))
    reader_1d.GetPointData().AddArray(array) # move points on caps slightly to ensure nice integration


ids = vtk.vtkIdList()
eps_norm = 1.0e-3 # integrate results on all points of intergration cells
print(f"Extracting solution at {reader_1d.GetNumberOfPoints()} points.")
only_caps = False # if True, only integrate at cap points, otherwise integrate at all points

for i in tqdm(range(reader_1d.GetNumberOfPoints())):
    # check if point is cap
    #print(branch_id[i])
    if branch_id[i] != 0:
        continue
    reader_1d.GetPointCells(i, ids)
    if ids.GetNumberOfIds() == 1:
        if gid[i] == 0:
            # inlet
            points[i] += eps_norm * normals[i]
        else:
            # outlets
            points[i] -= eps_norm * normals[i]
    else:
        if only_caps:
            continue # create integration object (slice geometry at point/normal)
    for time_str in times:
        #print(f"Time: {time_str}, Point: {i}, GID: {gid[i]}")
        if (int(time_str)%20 != 0):
            continue
        try:
            fpath_3d = f"/scratch/users/nrubio/synthetic_junctions/CCO/{tree_name}/{tree_name}_flow_unsteady/{num_procs}-procs/{tree_name}_unsteady_result_{time_str}.vtu"
            reader_3d = read_geo(fpath_3d).GetOutput()
            integral = get_integral(reader_3d, points[i], normals[i])
            reader_1d.GetPointData().GetArray(f'pressure_{time_str}').SetValue(i, integral.evaluate("Pressure"))
            reader_1d.GetPointData().GetArray(f'velocity_{time_str}').SetValue(i, integral.evaluate("Velocity"))

        except Exception:
            print("integration error")
            continue # integrate all output arrays

    try:
        fpath_3d = f"/scratch/users/nrubio/synthetic_junctions/CCO/{tree_name}/{tree_name}_flow_unsteady/{num_procs}-procs/{tree_name}_unsteady_result_{times[0]}.vtu"
        reader_3d = read_geo(fpath_3d).GetOutput()
        integral = get_integral(reader_3d, points[i], normals[i])
        reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    except:
        continue

write_geo(fpath_out, reader_1d)
print(f"wrote geo to {fpath_out}")
