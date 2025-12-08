
from collections import defaultdict
import json
import pdb
from unittest import result
import matplotlib
import casadi
import pandas as pd
import os
import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import save_dict

#from util.neural_net.nn_model import coef_loss
# Solve a zerod vascular flow with CasADi

# Load in svZeroDSolver input file
def solve_casadi_unsteady(input_file = None, result_df = None):

    num_vessels = len(input_file["vessels"])
    num_junctions = len(input_file["junctions"])
    num_BCs = len(input_file["boundary_conditions"])
    dt = input_file["boundary_conditions"][0]["bc_values"]["t"][1] - input_file["boundary_conditions"][0]["bc_values"]["t"][0]
    
    vessel_constraint_counter = 0
    junction_constraint_counter = 0
    BC_constraint_counter = 0
    SS_constraint_counter = 0
    coef_factor = 1

    # Create a CasADi Opti object
    opti = casadi.Opti()
    # Objective to minimize (squared residuals of the vessel and junction pressure equations)
    objective = 0

    inlet_Q = opti.parameter()
    # Decision variables
    Q_in = opti.variable(num_vessels)
    Q_out = opti.variable(num_vessels)
    Q_in_dt = opti.variable(num_vessels)
    Q_out_dt = opti.variable(num_vessels)

    P_in = opti.variable(num_vessels)
    P_out = opti.variable(num_vessels)
    P_in_dt = opti.variable(num_vessels)
    P_out_dt = opti.variable(num_vessels)

    inflow_extractors = opti.parameter(num_vessels, num_junctions)
    opti.set_value(inflow_extractors, 0)
    outflow_extractors = opti.parameter(num_vessels, num_junctions)
    opti.set_value(outflow_extractors, 0)

    # Dictionary to organize vessel information (needing for wiring junctions)
    vessel_dict = defaultdict(dict)
    bc_ind_dict = {}
    for ind, bc in enumerate(input_file["boundary_conditions"]):
        bc_ind_dict[bc["bc_name"]] = ind

    for i, vessel in enumerate(input_file["vessels"]):

        vessel_dict[vessel["vessel_id"]]["v_ind"] = i
        vessel_dict[vessel["vessel_id"]]["v_name"] = vessel["vessel_name"]

        R_lin = vessel["zero_d_element_values"]["R_poiseuille"]
        R_sten = vessel["zero_d_element_values"]["stenosis_coefficient"]#*0
        if "pressure_recovery_coefficient" in vessel["zero_d_element_values"].keys():
            R_quad = vessel["zero_d_element_values"]["pressure_recovery_coefficient"]
            print(f"R_quad: {R_quad} for vessel {vessel['vessel_name']}")
        else:
            R_quad = 0
        C = vessel["zero_d_element_values"]["C"]
        L = vessel["zero_d_element_values"]["L"]

        if "branch0" in vessel["vessel_name"]:
            
            objective += (
                P_in[i] +
                - P_out[i] +
                - (R_lin + R_sten * (10**-2 + Q_in[i]**2)**0.5 + R_quad * Q_in[i]) * Q_in[i] + # abs removed
                - L * Q_out_dt[i]  
                )**2
        else:
            opti.subject_to(
                P_out[i] + 
                - P_in[i] == 0
            )

        vessel_constraint_counter += 1
        
        # Conservation of mass (to satisfy exactly)
        opti.subject_to(
            Q_in[i] + 
            - Q_out[i] + 
            - C * P_in_dt[i] +
            C * (R_lin + 2*R_sten*(10**-10 + Q_in[i]**2)**0.5  + 2*R_quad*Q_in[i])*Q_in_dt[i] == 0 # abs removed
        )
        vessel_constraint_counter += 1
        
        # Outlet boundary conditions (to satisfy exactly)
        if "boundary_conditions" in vessel.keys():
            if "outlet" in vessel["boundary_conditions"].keys():
                bc_name = vessel["boundary_conditions"]["outlet"]
                bc_ind = bc_ind_dict[bc_name]
                resistance = input_file["boundary_conditions"][bc_ind]["bc_values"]["R"]
                opti.subject_to(P_out[i] - Q_out[i] * resistance == 0)
                BC_constraint_counter += 1
        
        # Inlet boundary conditions (to satisfy exactly)
            if "inlet" in vessel["boundary_conditions"].keys():
                opti.subject_to(Q_in[i] == inlet_Q)
                print(f"Inlet flow: {inlet_Q} for vessel {vessel['vessel_name']}")
                BC_constraint_counter += 1
                
    for i, junction in enumerate(input_file["junctions"]):
        j_name = junction["junction_name"]
        inlet_vessel_id = junction["inlet_vessels"][0]
        inlet_vessel_ind = vessel_dict[inlet_vessel_id]["v_ind"]
        outlet_vessel_inds = [vessel_dict[outlet_vessel]["v_ind"] for outlet_vessel in junction["outlet_vessels"]]

        for j, outlet_vessel_ind in enumerate(outlet_vessel_inds):

            if junction["junction_type"] == "BloodVesselJunction":

                R_lin = junction["junction_values"]["R_poiseuille"][j] * coef_factor
                if "inlet_R_lin" in junction["junction_values"].keys():
                    R_lin_inlet = junction["junction_values"]["inlet_R_lin"][0] * coef_factor
                else:
                    R_lin_inlet = 0
                R_sten = junction["junction_values"]["stenosis_coefficient"][j] * coef_factor
                if "pressure_recovery_coefficient" in junction["junction_values"].keys():
                    R_quad = junction["junction_values"]["pressure_recovery_coefficient"][j]
                else:
                    R_quad = 0
                print(f"R_quad: {R_quad} for junction {j_name} and outlet vessel {junction['outlet_vessels'][j]}")

                L = junction["junction_values"]["L"][j]
                C = 10^-8

                #Junction pressure equation residual (to minimize)
                objective += ((
                    P_out[inlet_vessel_ind] + # THIS IS THE INLET PRESSURE
                    - P_in[outlet_vessel_ind] +
                    - (R_lin + R_sten * (10**-2 + Q_in[outlet_vessel_ind]**2)**0.5 + R_quad * Q_in[outlet_vessel_ind]) * Q_in[outlet_vessel_ind] + # abs removed
                    - L * Q_in_dt[outlet_vessel_ind] +
                    - R_lin_inlet * Q_out_dt[inlet_vessel_ind])/(1333*inlet_Q**2)#(1333 *20)
                )**2
                junction_constraint_counter += 1
                
                enforce_pressure_loss = False
                if enforce_pressure_loss:
                    opti.subject_to(
                        P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] >= 0
                    )

                enforce_flow_splits = True
                if enforce_flow_splits:
                    objective += (
                        ((Q_in[outlet_vessel_ind] - junction["junction_values"]["flow_split"][j] *  Q_out[inlet_vessel_ind]))**2
                    )

            elif junction["junction_type"] == "NORMAL_JUNCTION":
                opti.subject_to(P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] == 0)
                junction_constraint_counter += 1

            
            opti.set_value(outflow_extractors[outlet_vessel_ind, i], -1)

        # Conservation of mass
        opti.set_value(inflow_extractors[inlet_vessel_ind, i], 1)
        opti.subject_to(Q_out.T@inflow_extractors[:,i] + Q_in.T@outflow_extractors[:,i] == 0)
        junction_constraint_counter += 1

    Q_in_prev = opti.parameter(num_vessels,1)
    Q_out_prev = opti.parameter(num_vessels,1)
    P_in_prev = opti.parameter(num_vessels,1)
    P_out_prev = opti.parameter(num_vessels,1)

        
    opti.subject_to(Q_in_dt    == (Q_in  - Q_in_prev)  / dt)
    opti.subject_to(Q_out_dt   == (Q_out - Q_out_prev) / dt)
    opti.subject_to(P_in_dt    == (P_in  - P_in_prev)  / dt)
    opti.subject_to(P_out_dt   == (P_out - P_out_prev) / dt)
    SS_constraint_counter += 4 * num_vessels

    # Enforce positive flows
    positive_flows = False
    if positive_flows:
        opti.subject_to(casadi.vec(Q_in)  >= 0)
        opti.subject_to(casadi.vec(Q_out) >= 0)
    
    # ---------------------------------------------------------------- #
    
    num_time_steps = int(len(input_file["boundary_conditions"][0]["bc_values"]["t"]))
    for time_step in range(num_time_steps):
        print(f"Solving time step {time_step + 1} of {num_time_steps}.")
        if time_step == 0:
            sol_prev = None
        time = input_file["boundary_conditions"][0]["bc_values"]["t"][time_step]
        
        print(f"Time step: {time_step}, Time: {time}, dt: {dt}")
        inlet_flow = input_file["boundary_conditions"][0]["bc_values"]["Q"][time_step]
        opti.set_value(inlet_Q, inlet_flow)
        # Enforce steady state
        if time_step == 0:
            opti.set_value(Q_in_prev, 0)
            opti.set_value(Q_out_prev, 0)
            opti.set_value(P_in_prev, 0)
            opti.set_value(P_out_prev, 0)
            
        else:
            # Steady state conditions
            opti.set_value(Q_in_prev, sol_prev["Q_in"])
            opti.set_value(Q_out_prev, sol_prev["Q_out"])
            opti.set_value(P_in_prev, sol_prev["P_in"])
            opti.set_value(P_out_prev, sol_prev["P_out"])
            
            opti.set_initial(Q_in, sol_prev["Q_in"]+dt*sol_prev["Q_in_dt"])
            opti.set_initial(Q_out, sol_prev["Q_out"]+dt*sol_prev["Q_out_dt"])
            opti.set_initial(Q_in_dt, sol_prev["Q_in_dt"])
            opti.set_initial(Q_out_dt, sol_prev["Q_out_dt"])

            opti.set_initial(P_in, sol_prev["P_in"]+dt*sol_prev["P_in_dt"])
            opti.set_initial(P_out, sol_prev["P_out"]+dt*sol_prev["P_out_dt"])
            opti.set_initial(P_in_dt, sol_prev["P_in_dt"])
            opti.set_initial(P_out_dt, sol_prev["P_out_dt"])
            
            

        # Solve NLP with IPOPT
        opti.minimize(objective) # Dummy objective
        #opti.solver('ipopt')
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver('ipopt', opts)
        try:
            sol = opti.solve()
            print("Objective value: ", opti.debug.value(objective))
        except:
            opti.debug.value(objective)
            print("Objective value: ", opti.debug.value(objective))
            opti.debug.value(Q_in)
            sol = opti.debug


        # Casadi solution to Pandas df
        num_vessels = len(vessel_dict.keys())
        for i, vessel in enumerate(vessel_dict.keys()):
            # pdb.set_trace()
            result_df.loc[i + time_step*num_vessels] = [vessel_dict[vessel]["v_name"],
                            time, # time?
                            sol.value(Q_in)[vessel_dict[vessel]["v_ind"]],
                            sol.value(Q_out)[vessel_dict[vessel]["v_ind"]],
                            sol.value(P_in)[vessel_dict[vessel]["v_ind"]],
                            sol.value(P_out)[vessel_dict[vessel]["v_ind"]]]
            
        sol_prev = {"Q_in": sol.value(Q_in),
                    "Q_out": sol.value(Q_out),
                    "P_in": sol.value(P_in),
                    "P_out": sol.value(P_out),
                    "Q_in_dt": sol.value(Q_in_dt),
                    "Q_out_dt": sol.value(Q_out_dt),
                    "P_in_dt": sol.value(P_in_dt),
                    "P_out_dt": sol.value(P_out_dt)}
        print("Pressure: ", sol.value(P_in)[0])
        
    return sol_prev



