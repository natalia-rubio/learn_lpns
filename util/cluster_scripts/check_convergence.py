import os
import sys
import math
import numpy as np
import pickle
import shutil
import pdb
import subprocess
import time
import copy
import get_avg_sol
import util.junction_proc
import centerline_proj

def check_convergence(geo_name, flow_index, anatomy, set_type, num_time_steps, inc):
    flow_name = f"flow_{flow_index}"
    results_dir = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{flow_index}"
    centerline_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/centerlines/centerline.vtp"
    print("Averaging 3D results.")
    print("Centerline dir: " + centerline_dir)
    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path = util.junction_proc.load_centerline_data(fpath_1d = centerline_dir)
    junction_dict, offsets, junc_pt_ids = util.junction_proc.identify_junctions_offset(junction_id, branch_id, pt_id, path, offset = 40)
    soln_dict, conv = get_avg_sol.get_avg_steady_results(ss_tol= 0.02, inc = inc,
                    fpath_1d = centerline_dir,
                    fpath_3d = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_{flow_index}_{int(num_time_steps):03d}.vtu",
                    fpath_3d_prev = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_{flow_index}_{int(num_time_steps)-inc:03d}.vtu",
                    fpath_out = results_dir,
                    pt_inds = junc_pt_ids, 
                    offsets = offsets,
                    only_caps=False)

    if conv == True:
        sys.exit(1)

    return

geo_name = sys.argv[1]; flow_index = sys.argv[2]; anatomy = sys.argv[3]; set_type = sys.argv[4];num_time_steps = sys.argv[5]; inc = int(sys.argv[6])
check_convergence(geo_name = geo_name, flow_index = flow_index, anatomy = anatomy, set_type = set_type, num_time_steps = num_time_steps, inc = inc)
