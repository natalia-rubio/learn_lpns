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

def export_time_series_offsets(geo_name, anatomy, set_type, num_time_steps, inc):
    flow_name = f"flow_unsteady"
    
    centerline_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/centerlines/centerline.vtp"
    #pdb.set_trace()
    if not os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}_new/{geo_name}"):
        os.makedirs(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}_new/{geo_name}")
        print("making new reduced results dictionary")
    #print("Averaging 3D results.")
    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path, points = util.junction_proc.load_centerline_data(fpath_1d = centerline_dir)
    #print("Extracted centerline data.")
    offset_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for percent_offset in offset_list:
        last_ts_done = True
        junction_dict = util.junction_proc.identify_junctions_percent_offset(junction_id, branch_id, pt_id, path, points, percent_offset)
        pdb.set_trace()
        results_dir = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}_new/{geo_name}/flow_unsteady_offset_{int(percent_offset*100)}_red_sol"
        print(f"Results dir: {results_dir}")
        if os.path.exists(results_dir):
            continue
        num_flows = int(num_time_steps/inc)
        master_soln_dict = {"times" : [], 
                            "flow_in_time" : [],
                            "pressure_in_time" : [],
                            "areas": [],
                            "tangents": [],
                            "lengths": [],
                            }
        for flow_index in range(num_flows):
            try:
                
                soln_dict = get_avg_sol.get_avg_unsteady_results(ss_tol= 0.02, inc = inc,
                            fpath_1d = centerline_dir,
                            fpath_3d = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_unsteady_{int((flow_index+1)*inc):03d}.vtu",
                            fpath_out = results_dir,
                            pt_inds = junction_dict[0]["branch_pts_offset"], 
                            offsets = junction_dict[0]["total_lengths"],
                            only_caps=False)
                for result_val in master_soln_dict.keys():
                    master_soln_dict[result_val].append(soln_dict[result_val])
            except:
                if flow_index == num_flows -1:
                    last_ts_done = False
                continue
                
        if len(master_soln_dict["times"]) > int(num_flows * 0.7) and last_ts_done:
            #pdb.set_trace()
            get_avg_sol.save_dict(master_soln_dict, results_dir)
        else:
            break

    return

geo_name = sys.argv[1]; flow_index = sys.argv[2]; anatomy = sys.argv[3]; set_type = sys.argv[4];num_time_steps = sys.argv[5]; inc = int(sys.argv[6])
if __name__ == "__main__":
    anatomy = sys.argv[1]
    set_type = sys.argv[2]
    num_time_steps = int(sys.argv[3])
    num_cores = int(sys.argv[4])
    num_geos = int(sys.argv[5])
    geo_start = int(sys.argv[6])
    inc = int(sys.argv[7])

    dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}"
    geos = os.listdir(dir); geos.sort(); geo_ind = 0;#
    #num_geos = len(geos)
    num_launched = 0

    while num_launched < num_geos:
        geo = geos[geo_ind+geo_start]; geo_name = geo; print(f"Geometry: {geo_name}")
        geo_ind += 1

        try:
            export_time_series_offsets(geo_name = geo_name, anatomy = anatomy, set_type = set_type, num_time_steps = num_time_steps, inc = inc)
        except Exception as error:
            # handle the exception
            print("An exception occurred:", type(error).__name__) 
            print(error)