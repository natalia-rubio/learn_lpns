import sys
sys.path.append("/home/users/nrubio/SV_scripts")
#from util.tools.basic import *

import vtk
import os
import numpy as np
import copy
from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm

from util.get_bc_integrals import get_res_names
import util.junction_proc
from util.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import pickle
#from sklearn.linear_model import LinearRegression

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def get_length(locs):
    length = 0
    for i in range(1, locs.shape[0]):
        length += np.linalg.norm(locs[i, :] - locs[i-1, :])
    return length


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
    for v in get_res_names(inp_3d, 'velocity'):
        fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        #fun = 'dot(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+',' + v + ")"
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)

def extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = 1000):

    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()# get all result array names
    res_names = get_res_names(reader_3d, ['pressure', 'velocity'])# get point and normals from centerline
    points = v2n(reader_1d.GetPoints().GetData())
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))# initialize output

    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
        array.Fill(0)
        reader_1d.GetPointData().AddArray(array) # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3 # integrate results on all points of intergration cells
    print(f"Extracting solution at {reader_1d.GetNumberOfPoints()} points.")
    for i in tqdm(range(reader_1d.GetNumberOfPoints())):
        # check if point is cap
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

        try:
            #import pdb; pdb.set_trace()
            integral = get_integral(reader_3d, points[i], normals[i])
        except Exception:
            continue # integrate all output arrays

        for name in res_names:
            reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
        reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    write_geo(fpath_out, reader_1d)
    return


# def plot_vars(anatomy, set_type, geometry, flow, plot_pressure = True, num_time_steps = 1000):
#     offset = 1
#     fpath_1dsol = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geometry}/1dsol_flow_solution_{flow}.vtp"
#     soln = read_geo(fpath_1dsol).GetOutput()  # get 3d flow data

#     soln_array = util.junction_proc.get_all_arrays(soln)
#     points = v2n(soln.GetPoints().GetData())
#     #Extract Geometry ----------------------------------------------------
#     pt_id = soln_array["GlobalNodeId"].astype(int)
#     num_pts = np.size(pt_id)  # number of points in mesh
#     branch_id = soln_array["BranchIdTmp"].astype(int)
#     junction_id = soln_array["BifurcationId"].astype(int)
#     inlet_pts = np.where(branch_id == 0)
#     outlet1_pts = np.where(branch_id == 1)
#     outlet2_pts = np.where(branch_id == 2)

#     zero_string = "0"
#     if len(num_time_steps) < 4:
#         zero_string = "00"

#     inlet_locs = (points[inlet_pts])[np.argsort(pt_id[inlet_pts])]
#     inlet_length = get_length(inlet_locs[offset:])
#     inlet_offset_length = get_length(inlet_locs[:offset])
#     p_inlet = (soln_array[f"pressure_{zero_string}{num_time_steps}"][inlet_pts])[np.argsort(pt_id[inlet_pts])]
#     p_end_inlet = p_inlet[offset]
#     q_inlet = (soln_array[f"velocity_{zero_string}{num_time_steps}"][inlet_pts])[np.argsort(pt_id[inlet_pts])][offset]
#     area_inlet = (soln_array["area"][inlet_pts])[np.argsort(pt_id[inlet_pts])][offset]
#     inlet_inds = np.linspace(0, len(p_inlet), len(p_inlet))
#     print(f"Inlet Pressure Difference: {np.max(p_inlet) - np.min(p_inlet)}")

#     outlet1_locs = (points[outlet1_pts])[np.argsort(pt_id[outlet1_pts])]
#     outlet1_length = get_length(outlet1_locs[:-offset])
#     outlet1_offset_length = get_length(outlet1_locs[-offset:])
#     p_outlet1 = (soln_array[f"pressure_{zero_string}{num_time_steps}"][outlet1_pts])[np.argsort(pt_id[outlet1_pts])]
#     p_end_outlet1 = p_outlet1[-offset]
#     q_outlet1 = (soln_array[f"velocity_{zero_string}{num_time_steps}"][outlet1_pts])[np.argsort(pt_id[outlet1_pts])][-offset]
#     area_outlet1 = (soln_array["area"][outlet1_pts])[np.argsort(pt_id[outlet1_pts])][-offset]
#     outlet1_inds = np.linspace(len(p_inlet), len(p_inlet)+len(p_outlet1), len(p_outlet1))

#     outlet2_locs = (points[outlet2_pts])[np.argsort(pt_id[outlet2_pts])]
#     outlet2_length = get_length(outlet2_locs[:-offset])
#     outlet2_offset_length = get_length(outlet2_locs[-offset:])
#     p_outlet2 = (soln_array[f"pressure_{zero_string}{num_time_steps}"][outlet2_pts])[np.argsort(pt_id[outlet2_pts])]
#     p_end_outlet2 = p_outlet2[-offset]
#     q_outlet2 = (soln_array[f"velocity_{zero_string}{num_time_steps}"][outlet2_pts])[np.argsort(pt_id[outlet2_pts])][-offset]
#     area_outlet2 = (soln_array["area"][outlet2_pts])[np.argsort(pt_id[outlet2_pts])][-offset]
#     outlet2_inds = np.linspace(len(p_inlet), len(p_inlet)+len(p_outlet2), len(p_outlet2))

#     reg_in = LinearRegression().fit(inlet_inds[2*offset: -offset].reshape(-1, 1), p_inlet[2*offset: -offset].reshape(-1, 1))
#     p_inlet_pred  = reg_in.predict(inlet_inds.reshape(-1, 1))


#     reg1 = LinearRegression().fit(outlet1_inds[-220: -offset].reshape(-1, 1), p_outlet1[-220: -offset].reshape(-1, 1))
#     p_outlet1_pred  = reg1.predict(outlet1_inds.reshape(-1, 1))
#     dp_junc1 = p_outlet1_pred[0] - p_inlet_pred[-1]

#     reg2 = LinearRegression().fit(outlet2_inds[-220: -offset].reshape(-1, 1), p_outlet2[-220: -offset].reshape(-1, 1))
#     p_outlet2_pred  = reg2.predict(outlet2_inds.reshape(-1, 1))
#     dp_junc2 = p_outlet2_pred[0] - p_inlet_pred[-1]

#     #print(f"PRESSURE || inlet: {p_inlet}.  outlet_1: {p_outlet1}. outlet_2: {p_outlet2}")
#     #assert area_inlet > area_outlet1, "Outlet1 area larger than inlet area"
#     #assert area_inlet > area_outlet2, "Outlet2 area larger than inlet area"
#     assert (q_inlet - (q_outlet1 +  q_outlet2))/q_inlet < 0.02, "Flow not conserved"

#     soln_dict = {"flow": np.asarray([q_outlet1, q_outlet2, q_inlet]),
#                 "dp_junc": np.asarray([dp_junc1[0], dp_junc2[0]]),
#                 "dp_end": np.asarray([p_end_outlet1 - p_end_inlet, p_end_outlet2 - p_end_inlet]),
#                 "area": np.asarray([area_outlet1, area_outlet2, area_inlet]),
#                 "length": np.asarray([outlet1_length, outlet2_length, inlet_length])}

#     if soln_dict["area"][0] < soln_dict["area"][1]:
#         print(f"Switching outlets on {geometry}.")
#         for value in soln_dict.keys():
#             tmp = copy.deepcopy(soln_dict[value][0])
#             soln_dict[value][0] = soln_dict[value][1]
#             soln_dict[value][1] = tmp
#     save_dict(soln_dict, f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geometry}/flow_{flow}_red_sol_full")
#     return
