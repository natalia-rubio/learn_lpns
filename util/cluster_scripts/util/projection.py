# Copyright (c) Stanford University, The Regents of the University of
#               California, and others.
#
# All Rights Reserved.
#
# See Copyright-SimVascular.txt for additional details.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

"""

import os
import re
import csv
import glob
import logging
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

# from .manage import get_logger_name
# from .solver import Solver
try:
    from util.geometry_generation.initialization_helpers.vtk_functions_API import read_geo, write_geo, add_array, collect_arrays, ClosestPoints, region_grow
except:
    from util.vtk_functions_API import read_geo, write_geo, add_array, collect_arrays, ClosestPoints, region_grow
import pdb

# get rid of numpy warnings
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def project_0d_to_3D(anatomy, set_type, geo_name, flow_index, local = False):
    print("Entered projection function.")
    if local == False:
        sher_dir = "/scratch/users/nrubio"
        fpath_0d = sher_dir + "/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/flow_"+str(flow_index)+"/zerod_files/zerod_soln.csv"
        fpath_3d = sher_dir + "/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/mesh-complete/mesh-complete.mesh.vtu"
        fpath_walls = sher_dir + "/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/mesh-complete/walls_combined.vtp"
        fpath_cent = sher_dir + "/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/centerlines/centerline.vtp"
        fpath_cent_proj = sher_dir + "/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name +"/flow_"+str(flow_index)+"/0D_cent_proj.vtp"
        fpath_init_soln = sher_dir + "/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/flow_"+str(flow_index)+"/initial_soln.vtu"
        
    else:
        fpath_0d = "data/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/flow_"+str(flow_index)+"/zerod_files/zerod_soln.csv"
        fpath_3d = "data/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/mesh-complete/mesh-complete.mesh.vtu"
        fpath_walls = "data/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/mesh-complete/walls_combined.vtp"
        fpath_cent = "data/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/centerlines/centerline.vtp"
        fpath_cent_proj = "data/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name +"/flow_"+str(flow_index)+"/0D_cent_proj.vtp"
        fpath_init_soln = "data/synthetic_junctions/"+anatomy+"/"+set_type+"/"+geo_name+"/flow_"+str(flow_index)+"/initial_soln.vtu"

    print("Reading 0D results from: " + fpath_0d)
    results_0d, final_time = read_results_0d(fpath_0d)
    print("Projecting 0D results to centerline")
    project_results_to_centerline(results_0d, final_time, fpath_cent, fpath_cent_proj)
    print("Projecting centerline results to 3D mesh")
    project_centerline_to_3d(fpath_3d, fpath_cent_proj, fpath_walls, fpath_init_soln)
    return

def read_results_0d(zerod_soln_path):
    """
    Read results from svZeroDSolver and store in dictionary:
    results_0d[result field][branch id][segment id][time step]
    """
    res_full = np.loadtxt(zerod_soln_path, delimiter=",", dtype=str)
    def rec_dict():
        """
        Recursive defaultdict
        """
        return defaultdict(rec_dict)
    results_0d = defaultdict(rec_dict)
    locations = res_full[1:, 0]
    times = res_full[1:, 1]
    
    field_list = ["flow_in", "flow_out", "pressure_in", "pressure_out"]
    for field_ind, field in enumerate(field_list):
        for i in range(len(locations)):
            location = locations[i]
            branchID = int(location[6])
            segID = int(location[11])
            timestep = float(times[i])
            #print("timestep: ", timestep)
            if timestep > 0.7:
                results_0d[field][branchID][segID][timestep] = float(res_full[i+1, field_ind+2])
                final_time = timestep

    return results_0d, final_time

def project_results_to_centerline(results_0d, final_time, fpath_cent, fpath_cent_proj):
    """
    Project rom results onto the centerline
    """
    # assemble output dict
    rec_dd = lambda: defaultdict(rec_dd)
    arrays = rec_dd()

    cent = read_geo(fpath_cent)
    # extract point arrays from geometries
    arrays_cent = collect_arrays(cent.GetPointData())
    for name, data in arrays_cent.items():
        arrays[name] = data

    # centerline points
    points = v2n(cent.GetPoints().GetData())

    # all branch ids in centerline
    ids_cent = np.unique(arrays_cent['BranchId']).tolist()
    ids_cent.remove(-1)

    # loop all result fields
    for f in ["pressure", "flow"]:
        # check if ROM branch has same ids as centerline
        ids_rom = list(results_0d[f+"_in"].keys())
        ids_rom.sort()
        print("ids_cent: ", ids_cent)
        print("ids_rom: ", ids_rom)
        assert ids_cent == ids_rom, 'Centerline and ROM results have different branch ids'

        # initialize output arrays
        array_f = np.zeros((arrays_cent['Path'].shape[0], 1))
        n_outlet = np.zeros(arrays_cent['Path'].shape[0])

        # loop all branches
        for br in results_0d[f+"_in"].keys():

            segs = results_0d[f+"_in"][br].keys()
            min_seg = min(segs)
            max_seg = max(segs)
            # results of this branch
            res_br = np.asarray([results_0d[f+"_in"][br][min(segs)][final_time],
                                results_0d[f+"_out"][br][max(segs)][final_time]])

            # get centerline path
            path_cent = arrays_cent['Path'][arrays_cent['BranchId'] == br]

            # get node locations from 0D results
            path_1d_res = path_cent # results_0d['distance'][br]
            f_res = res_br
            # print("1D res path:")
            # print(path_1d_res.keys())
            # assert np.isclose(path_1d_res[0], 0.0), 'ROM branch path does not start at 0'
            # assert np.isclose(path_cent[0], 0.0), 'Centerline branch path does not start at 0'
            # msg = 'ROM results and centerline have different branch path lengths'
            # assert np.isclose(path_1d_res[-1], path_cent[-1]), msg

            # interpolate ROM onto centerline
            # limit to interval [0,1] to avoid extrapolation error interp1d due to slightly incompatible lenghts
            f_cent_int = interp1d(np.asarray([0, 1]), np.asarray([f_res[0], f_res[1]])) #path_1d_res / path_1d_res[-1], f_res.T)(path_cent / path_cent[-1]).T
            f_cent = f_cent_int((path_cent - min(path_cent)) / (max(path_cent) - min(path_cent)))
            # store results of this path
            array_f[arrays_cent['BranchId'] == br] = f_cent.reshape(-1, 1)

            # add upstream part of branch within junction
            if br == 0:
                continue

            # first point of branch
            ip = np.where(arrays_cent['BranchId'] == br)[0][0]

            # centerline that passes through branch (first occurence)
            cid = np.where(arrays_cent['CenterlineId'][ip])[0][0]

            # id of upstream junction
            jc = arrays_cent['BifurcationId'][ip - 1]

            # centerline within junction
            is_jc = arrays_cent['BifurcationId'] == jc
            jc_cent = np.where(np.logical_and(is_jc, arrays_cent['CenterlineId'][:, cid]))[0]

            # length of centerline within junction
            jc_path = np.append(0, np.cumsum(np.linalg.norm(np.diff(points[jc_cent], axis=0), axis=1)))
            jc_path /= jc_path[-1]

            # results at upstream branch
            res_br_u = results_0d[f+"_out"][arrays_cent['BranchId'][jc_cent[0] - 1]]

            # results at beginning and end of centerline within junction
            f0 = res_br_u[0][final_time]
            f1 = res_br[0]
            # map 1d results to centerline using paths
            array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T)(jc_path).T

            # count number of outlets of this junction
            n_outlet[jc_cent] += 1
        # normalize results within junctions by number of junction outlets
        is_jc = n_outlet > 0
        array_f[is_jc] = (array_f[is_jc].T / n_outlet[is_jc]).T

        # assemble time steps
        for i, t in enumerate([final_time,]):
            arrays[f] = array_f[:, i]

    # add arrays to centerline and write to file
    for f, a in arrays.items():
        out_array = n2v(a)
        out_array.SetName(f)
        cent.GetPointData().AddArray(out_array)
    write_geo(fpath_cent_proj, cent)

def get_centerline_3d_map(geo3d, geo1d):
    """
    Create a map from centerine to volume mesh through region growing
    """
    # get points
    
    points_vol = v2n(geo3d.GetPoints().GetData())
    points_1d  = v2n(geo1d.GetPoints().GetData())

    # get volume points closest to centerline
    cp_vol = ClosestPoints(geo3d)
    seed_points = np.unique(cp_vol.search(points_1d))

    # map centerline points to selected volume points
    cp_1d = ClosestPoints(geo1d)
    seed_ids = np.array(cp_1d.search(points_vol[seed_points]))

    # call region growing algorithm
    ids, dist, rad = region_grow(geo3d, seed_points, seed_ids, n_max=999)

    # check 1d to 3d map
    assert np.max(ids) <= geo1d.GetNumberOfPoints() - 1, '1d-3d map non-conforming'

    return ids, dist, rad


def project_centerline_to_3d(fpath_3d, fpath_cent_proj, fpath_walls, fpath_init_soln):
    """
    Map 1D results on centerline to volume mesh
    """
    geo3d = read_geo(fpath_3d); geo1d = read_geo(fpath_cent_proj); geowalls = read_geo(fpath_walls)
    # get 1d -> 3d map
    map_ids, map_iter, map_rad = get_centerline_3d_map(geo3d, geo1d)

    # get arrays
    arrays_cent = collect_arrays(geo1d.GetPointData())

    # map all centerline arrays to volume geometry
    for name, array in arrays_cent.items():
        add_array(geo3d, name, array[map_ids])

    # add mapping to volume mesh
    for name, array in zip(['MapIds', 'MapIters'], [map_ids, map_iter]):
        add_array(geo3d, name, array)

    # inverse map
    map_ids_inv = {}
    for i in np.unique(map_ids):
        map_ids_inv[i] = np.where(map_ids == i)

    # create radial coordinate [0, 1]
    rad = np.zeros(geo3d.GetNumberOfPoints())
    for i, ids in map_ids_inv.items():
        rad_max = np.max(map_rad[ids])
        if rad_max == 0:
            rad_max = np.max(map_rad)
        rad[ids] = map_rad[ids] / rad_max
    add_array(geo3d, 'rad', rad)

    # set points at wall to hard 1
    wall_ids = collect_arrays(geowalls.GetPointData())['GlobalNodeID'].astype(int) - 1
    rad[wall_ids] = 1

    # mean velocity
    for a in arrays_cent.keys():
        if 'flow' in a:
            u_mean = arrays_cent[a] / arrays_cent['CenterlineSectionArea']

            # parabolic velocity
            u_quad = 2 * u_mean[map_ids] * (1 - rad ** 2)

            # scale parabolic flow profile to preserve mean flow
            for i, ids in map_ids_inv.items():
                u_mean_is = np.mean(u_quad[map_ids_inv[i]])
                u_quad[ids] *= u_mean[i] / u_mean_is

            # parabolic velocity vector field
            velocity = np.outer(u_quad, np.ones(3)) * arrays_cent['CenterlineSectionNormal'][map_ids]

            # add to volume mesh
            add_array(geo3d, a.replace('flow', 'velocity'), velocity)

    # write to file
    write_geo(fpath_init_soln, geo3d)

# class Post(object):
#     """
#     Does all the postprocessing (0D or 1D):
#       - read results
#       - project results onto centerline
#       - project centerline onto 3D mesh
#     """
#     def __init__(self, params, logger):
#         self.params = params
#         self.logger = logger

#         # always read all segments
#         self.params.all_segments = True

#         # initialize results [result field][branch id][segment id][time step]
#         self.results = {}

#         geometries = {'1d': self.params.oned_model,
#                       'cent': self.params.centerlines_file,
#                       'surf': self.params.walls_mesh_file,
#                       'vol': self.params.volume_mesh_file}

#         self.geos = {}
#         for name, pth in geometries.items():
#             self.geos[name] = None
#             if pth is not None:
#                 fpath = os.path.join(self.params.results_directory, pth)
#                 if os.path.exists(fpath):
#                     self.geos[name] = read_geo(fpath)

#     def process(self):
#         """
#         Do all the post-processing
#         (depending on model order and what results/geometries are given)
#         """
#         # read results
#         if self.params.model_order == 0:
#             self.read_results_0d()
#         elif self.params.model_order == 1:
#             self.read_results_1d()

#         # get time steps to process
#         self.get_time_steps()

#         # write results to numpy file
#         self.write_results()

#         # all files given for projection?
#         project = self.params.model_order == 0 or (self.params.model_order == 1 and self.geos['1d'] is not None)

#         # project results to centerline
#         if project and self.geos['cent'] is not None:
#             self.project_results_to_centerline()

#         # project centerline to 3D mesh
#         if project and self.geos['surf'] is not None and self.geos['vol'] is not None:
#             self.project_centerline_to_3d()

#     def get_time_steps(self):
#         """
#         Select time steps to retrieve from results
#         """
#         if self.params.model_order == 0:
#             # time steps given in results
#             lower = self.results['time'] >= self.params.time_range[0]
#             upper = self.results['time'] <= self.params.time_range[1]
#             self.params.time_indices = np.where(np.logical_and(lower, upper))[0]
#             self.params.times = self.results['time'][self.params.time_indices]
#         elif self.params.model_order == 1:
#             # time steps retrieved in Solver class
#             pass



#     def read_results_1d(self):
#         """
#         Read results from svOneDSolver and store in dictionary:

#         results_1d[result field][branch id][discretization node, time step]

#         time[time step]
#         distance[branch id][discretization node]
#         """
#         for field in self.params.data_names:
#             # file name pattern for 1D results
#             out_name = '*branch*seg*_' + field + '.dat'

#             # list all output files for field
#             result_list_1d = glob.glob(os.path.join(self.params.results_directory, out_name))

#             # loop segments
#             self.results[field] = defaultdict(dict)
#             for f_res in result_list_1d:
#                 with open(f_res) as f:
#                     reader = csv.reader(f, delimiter=' ')

#                     # loop nodes
#                     results_1d_f = []
#                     for line in reader:
#                         results_1d_f.append([float(l) for l in line if l][1:])

#                 # store results and GroupId
#                 seg = int(re.findall(r'\d+', f_res)[-1])
#                 branch = int(re.findall(r'\d+', f_res)[-2])
#                 self.results[field][branch][seg] = np.array(results_1d_f)

#     def write_results(self):
#         """
#         Write rom results as dictionary of numpy arrays to file
#         """
#         f_out = os.path.join(self.params.output_directory, self.params.output_file_name + '.npy')
#         np.save(f_out, self.results)





#     def project_centerline_to_3d(self):
#         """
#         Map 1D results on centerline to volume mesh
#         """
#         # get 1d -> 3d map
#         map_ids, map_iter, map_rad = self.get_centerline_3d_map()

#         # get arrays
#         arrays_cent = collect_arrays(self.geos['cent'].GetPointData())

#         # map all centerline arrays to volume geometry
#         for name, array in arrays_cent.items():
#             add_array(self.geos['vol'], name, array[map_ids])

#         # add mapping to volume mesh
#         for name, array in zip(['MapIds', 'MapIters'], [map_ids, map_iter]):
#             add_array(self.geos['vol'], name, array)

#         # inverse map
#         map_ids_inv = {}
#         for i in np.unique(map_ids):
#             map_ids_inv[i] = np.where(map_ids == i)

#         # create radial coordinate [0, 1]
#         rad = np.zeros(self.geos['vol'].GetNumberOfPoints())
#         for i, ids in map_ids_inv.items():
#             rad_max = np.max(map_rad[ids])
#             if rad_max == 0:
#                 rad_max = np.max(map_rad)
#             rad[ids] = map_rad[ids] / rad_max
#         add_array(self.geos['vol'], 'rad', rad)

#         # set points at wall to hard 1
#         wall_ids = collect_arrays(self.geos['surf'].GetPointData())['GlobalNodeID'].astype(int) - 1
#         rad[wall_ids] = 1

#         # mean velocity
#         for a in arrays_cent.keys():
#             if 'flow' in a:
#                 u_mean = arrays_cent[a] / arrays_cent['CenterlineSectionArea']

#                 # parabolic velocity
#                 u_quad = 2 * u_mean[map_ids] * (1 - rad ** 2)

#                 # scale parabolic flow profile to preserve mean flow
#                 for i, ids in map_ids_inv.items():
#                     u_mean_is = np.mean(u_quad[map_ids_inv[i]])
#                     u_quad[ids] *= u_mean[i] / u_mean_is

#                 # parabolic velocity vector field
#                 velocity = np.outer(u_quad, np.ones(3)) * arrays_cent['CenterlineSectionNormal'][map_ids]

#                 # add to volume mesh
#                 add_array(self.geos['vol'], a.replace('flow', 'velocity'), velocity)

#         # write to file
#         f_out = os.path.join(self.params.output_directory, self.params.output_file_name + '.vtu')
#         write_geo(f_out, self.geos['vol'])


# def res_rom_to_path(path, res):
#     """
#     Map 0d and 1d results to vessel path
#     """
#     path_1d = []
#     int_1d = []
#     for seg, res_1d_seg in sorted(res.items()):
#         # 1d results are duplicate at FE-nodes at corners of segments
#         if seg == 0:
#             # start with first FE-node
#             i_start = 0
#         else:
#             # skip first FE-node (equal to last FE-node of previous segment)
#             i_start = 1

#         # generate path for segment FEs, assuming equidistant spacing
#         p0 = path[seg]
#         p1 = path[seg + 1]
#         path_1d += np.linspace(p0, p1, res_1d_seg.shape[0])[i_start:].tolist()
#         int_1d += res_1d_seg[i_start:].tolist()

#     return np.array(path_1d), np.array(int_1d)
