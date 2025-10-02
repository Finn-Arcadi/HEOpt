import optFunction as optF
import numpy as np

inputObj = optF.inputConstructor() #set up object

#USER INPUTS--------
#General
inputObj.shell_inlet_temp = 35.0 #C
inputObj.tube_inlet_temp = 75.0 #C
inputObj.OD_tube = 0.003175 #m - 0.003175m = 0.115in
inputObj.wall_tube = .000127 #m
inputObj.heat_transfer = 20000 #W
inputObj.U_penalty = 0 #penalty on overall heat transfer coefficient U from 0 to 1, 0 = no penalty, 1 = no heat transfer
inputObj.staggered = 0 #1 = staggered, 0 = inline

#Material and Environment
inputObj.shell_ref_pressure = 101325.0 #Pa
inputObj.tube_ref_pressure = 210000.0 #Pa
inputObj.relative_humidity = 0.0 # from 0 to 100
inputObj.EGW_percentage = 50 # Ethylene glycol content in water from 0 to 100
inputObj.tube_absolute_roughness = 0.000005; #m, absolute roughness of inner surface of tubes;  #5 micron roughness for steel
inputObj.rho_tube = 7850 #kg/m^3, density of tube material; 7850 for steel
inputObj.k_tube = 25 #W/mK, conductivity of tube material; ~25 for steel

#Contraint Variables
inputObj.dT_t_max = 10 #C
inputObj.core_height_max = .3048 #m
inputObj.core_length_max = 0.4572 #m
inputObj.core_depth_max = 0.0793 #.127 #m
inputObj.min_effectiveness = 0.75
inputObj.max_dP_t_psi = 10 #psi
inputObj.max_dP_s_Pa = 150 #Pa
inputObj.min_NT = 10
inputObj.min_NL = 10 #must be 10 for valid model
inputObj.max_NT = None
inputObj.max_NL = None
inputObj.min_ST_D = 1.25
inputObj.min_SL_D = 1.25
inputObj.max_ST_D = 5
inputObj.max_SL_D = 3

#Run the optimization problem
print('\n RUNNING OPTIMIZATION PROBLEM \n')
probOutput = optF.evalopt(inputObj, 'opt')
#optF.printOutput(probOutput)

#Variables required for final evaluation:
inputObj.T_out_ts = probOutput.get_val('top_group.CG.HT.T_out_t')[0] #K
inputObj.T_out_ss = probOutput.get_val('top_group.CG.HT.T_out_s')[0] #K
inputObj.SL_D = probOutput.get_val('top_group.CG.HT.SL_D')[0] #m
inputObj.ST_D = probOutput.get_val('top_group.CG.HT.ST_D')[0] #m
inputObj.NL = np.ceil(probOutput.get_val('top_group.CG.NL')[0])
inputObj.NT = np.ceil(probOutput.get_val('top_group.CG.NT')[0])
inputObj.L_tube = probOutput.get_val('top_group.CG.HT.L_tube')[0] #m
inputObj.OD_tube = probOutput.get_val('top_group.CG.HT.OD_tube')[0] #m
inputObj.wall_tube = probOutput.get_val('top_group.CG.HT.wall_tube_output')[0] #m

print('\n EVALUATING OUTPUTS OF OPTIMIZATION PROBLEM \n')
evalOutput = optF.evalopt(inputObj, 'eval')

optF.printOutput(evalOutput)
optF.getCSV(evalOutput, '20kW-LF-Inline')