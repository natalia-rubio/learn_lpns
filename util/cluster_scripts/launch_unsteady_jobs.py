from util.sherlock_util import *
# from util.svFSI.util.projection import * For solution initialization

set_type = sys.argv[1]
start_ind = int(sys.argv[2])
num_geos = int(sys.argv[3])

time_step_size = 0.001
num_launched = 0

anatomy = "CCO_trees"
set_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}"
print(set_dir)
geos = os.listdir(set_dir); geos.sort(); geo_ind = start_ind;#
print(geos)
while num_launched < num_geos:

    geo = geos[geo_ind]; geo_name = geo; print(f"Geometry: {geo_name}")
    geo_ind += 1
    dir = set_dir + f"/{geo_name}"
    print(geo_name)
    if not check_geo_name(geo):
        continue
    
    if not check_for_centerline(anatomy, set_type, geo_name):
        continue
    if os.path.exists(f"{dir}/48-procs/{geo_name}_unsteady_result_700.vtu"):
        continue
    print("Found centerline")
    os.system(f"cd {dir}/ && sbatch {geo_name}.sh")
    
    print(f"Started job for {geo} unsteady flow")
    print("\n\
        ---------------------------------\n")    

            #except:
                #continue
    num_launched +=1
