import numpy as np
import openmdao.api as om
from radiator_model import CycleGeometry, DarcyFrictionFactorComp, PressureLoss
from scipy import constants
import csv

def evalopt(inputObj, mode):

    #Prelim calculations
    #Convert temperatures to K
    T_in_s = constants.convert_temperature(inputObj.shell_inlet_temp, 'C', 'K')
    T_in_t = constants.convert_temperature(inputObj.tube_inlet_temp, 'C', 'K')

    prob = om.Problem()
    model = prob.model

    #Top level group for full dynamics
    top_group = model.add_subsystem('top_group', om.Group())
    top_group.add_subsystem('CG', CycleGeometry())
    top_group.add_subsystem('DFF', DarcyFrictionFactorComp())
    top_group.add_subsystem('PL', PressureLoss())

    #Make connections for darcy friction factor:
    top_group.connect('CG.HT.Re_D_t', 'DFF.Re_D_t')
    top_group.connect('CG.HT.ID_tube', 'DFF.ID_tube')

    #Make connections for pressure loss
    top_group.connect('DFF.f', 'PL.Darcy_f')
    top_group.connect('CG.HT.V_t', 'PL.V_t')
    top_group.connect('CG.SA.L_tube_req', 'PL.Length')
    top_group.connect('CG.HT.ID_tube', 'PL.ID_tube')
    top_group.connect('CG.HT.rho_t', 'PL.rho_t')

    #Set solvers for Darcy Friction Factor

    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.nonlinear_solver.linesearch.options['print_bound_enforce'] = True

    model.linear_solver = om.DirectSolver()
    model.linear_solver.options['rhs_checking'] = True

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 1000
    prob.driver.options['tol'] = 1e-6

    prob.set_solver_print(level=0)


    #Design variables - if optimizing
    if mode == 'opt':

        #Pre-calculate constraint temperatures
        T_out_t_min = max(T_in_t - inputObj.dT_t_max, 1.0001*(T_in_s)) #Hot side outlet must respect dT and be greater than cold side inlet
        T_out_t_max = .9999 * T_in_t #Ensures there is always heat transfer so gradients can solve
        T_out_s_min = 1.0001 * T_in_s #Ensures there is always heat transfer so gradients can solve
        T_out_s_max = .9999 * T_in_t #Ensures there is always heat transfer so gradients can solve

        prob.model.add_design_var('top_group.CG.T_out_t', lower = T_out_t_min, upper = T_out_t_max)
        prob.model.add_design_var('top_group.CG.T_out_s', lower = T_out_s_min, upper = T_out_s_max)
        
        prob.model.add_design_var('top_group.CG.SL_D', lower = inputObj.min_SL_D, upper= inputObj.max_SL_D)
        prob.model.add_design_var('top_group.CG.ST_D', lower = inputObj.min_ST_D , upper = inputObj.max_ST_D)
        
        prob.model.add_design_var('top_group.CG.NL', lower = inputObj.min_NL, upper=inputObj.max_NL)
        prob.model.add_design_var('top_group.CG.NT', lower = inputObj.min_NT, upper=inputObj.max_NT)

        #Objective
        prob.model.add_objective('top_group.CG.core_wet_mass')

        #Constraints
        prob.model.add_constraint('top_group.CG.HT.L_tube', lower = 0, upper=inputObj.core_length_max)
        prob.model.add_constraint('top_group.CG.H_core', upper=inputObj.core_height_max)
        prob.model.add_constraint('top_group.CG.D_core', upper=inputObj.core_depth_max)
        prob.model.add_constraint('top_group.PL.dP_maj_t', upper = inputObj.max_dP_t_psi * 6895) #approx conversion for Pa to psi
        prob.model.add_constraint('top_group.CG.dP_maj_s', upper = inputObj.max_dP_s_Pa)
        prob.model.add_constraint('top_group.CG.NTU_effectiveness', lower = inputObj.min_effectiveness)

        # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
        prob.model.approx_totals()

        prob.setup()

        #Set initial values
        prob.model.set_val('top_group.CG.T_out_t', T_out_t_min, units='K')
        prob.model.set_val('top_group.CG.T_out_s', T_out_s_max, units='K')
        prob.model.set_val('top_group.CG.SL_D', inputObj.min_SL_D)
        prob.model.set_val('top_group.CG.ST_D', inputObj.max_ST_D)
        prob.model.set_val('top_group.CG.NL', 50)
        prob.model.set_val('top_group.CG.NT', 50)
        prob.model.set_val('top_group.CG.OD_tube', inputObj.OD_tube)

    elif mode == 'eval':

        prob.setup()

        #Instead set the design vars here


        prob.set_val('top_group.CG.T_out_t', inputObj.T_out_ts, units='K')
        prob.set_val('top_group.CG.T_out_s', inputObj.T_out_ss, units='K')
        prob.set_val('top_group.CG.HT.SL_D', inputObj.SL_D)
        prob.set_val('top_group.CG.HT.ST_D', inputObj.ST_D)
        prob.set_val('top_group.CG.NL', inputObj.NL)
        prob.set_val('top_group.CG.NT', inputObj.NT)
        prob.set_val('top_group.CG.HT.L_tube', inputObj.L_tube, units='m')

        prob.set_val('top_group.CG.OD_tube', inputObj.OD_tube, units='m')

    else:
        print('input variable "mode" must be either "opt" to optimize design or "eval" to evaluate performance at inputs')

    #General (constant) user inputs
    
    prob.set_val('top_group.CG.wall_tube', inputObj.wall_tube, units='m')

    prob.set_val('top_group.CG.Qdot', inputObj.heat_transfer, units='W')
    prob.set_val('top_group.CG.T_in_t', T_in_t, units='K')
    prob.set_val('top_group.CG.T_in_s', T_in_s, units='K')
    prob.set_val('top_group.CG.k_tube', inputObj.k_tube, units='W/(m*K)')
    prob.set_val('top_group.CG.rho_tube', inputObj.rho_tube, units='kg/m**3')
    prob.set_val('top_group.DFF.epsilon', inputObj.tube_absolute_roughness, units='m')
    prob.set_val('top_group.CG.U_penalty', inputObj.U_penalty)
    prob.set_val('top_group.CG.HT.staggered', inputObj.staggered)
    prob.set_val('top_group.CG.HT.Shell_Ref_Pressure', inputObj.shell_ref_pressure)
    prob.set_val('top_group.CG.HT.Tube_Ref_Pressure', inputObj.tube_ref_pressure)
    prob.set_val('top_group.CG.HT.Relative_Humidity', inputObj.relative_humidity)
    prob.set_val('top_group.CG.HT.EGW_percentage', inputObj.EGW_percentage)


    # Choose whether to evaluate the performance at the given inputs or optimize the design
    if mode == 'eval':
        prob.run_model()
    elif mode == 'opt':
        prob.run_driver()
    else:
        print('input variable "mode" must be either "opt" to optimize design or "eval" to evaluate performance at inputs')

    return prob

def printOutput(printObj):

    #Heat transfer coefficients 
    print('\n Heat transfer achieved via NTU estimate (W): ', round(printObj.get_val('top_group.CG.Q_dot_NTU')[0], 3), ' | Heat transfer achieved via LMTD estimate (W): ', round(printObj.get_val('top_group.CG.Q_dot_LMTD')[0], 3))
    print('\n Overall heat transfer coefficient U (W/m^2K): ', round(printObj.get_val('top_group.CG.U')[0], 3),
          ' | Tube-side convective heat transfer coefficient h_t (W/m^2K): ', round(printObj.get_val('top_group.CG.h_t')[0], 3), ' | Shell-side convective heat transfer coefficient h_s (W/m^2K): ', round(printObj.get_val('top_group.CG.h_s')[0], 3),
          ' | Tube thermal resistance R_tube (m^2K/W): ', round(printObj.get_val('top_group.CG.R_tube')[0], 3), ' | Heat transfer surface Area (m^2): ', round(printObj.get_val('top_group.CG.HT.A_req')[0], 3))
    
    # Fluid Properties
    print('\n cP_s (J/kgK): ', round(printObj.get_val('top_group.CG.HT.cP_s')[0], 3), ' | cP_t (J/kgK): ', round(printObj.get_val('top_group.CG.HT.cP_t')[0], 3))
    print('\n k_s (W/mK): ', round(printObj.get_val('top_group.CG.HT.k_s')[0], 3), ' | k_t (W/mK): ', round(printObj.get_val('top_group.CG.HT.k_t')[0], 3))
    print('\n rho_s (kg/m^3): ', round(printObj.get_val('top_group.CG.HT.rho_s')[0], 3), ' | rho_t (kg/m^3): ', round(printObj.get_val('top_group.CG.HT.rho_t')[0], 3))
    print('\n mu_s (mPa*s): ', round(printObj.get_val('top_group.CG.HT.mu_s')[0]*1000, 8), ' | mu_t (mPa*s): ', round(printObj.get_val('top_group.CG.HT.mu_t')[0]*1000, 3))
    
    # Reynolds, Prandtl, and Nusselt Numbers
    print('\n Re_D_s: ', round(printObj.get_val('top_group.CG.HT.Re_D_s')[0], 3), ' | Re_D_t: ', round(printObj.get_val('top_group.CG.HT.Re_D_t')[0], 3))
    print('\n Pr_s: ', round(printObj.get_val('top_group.CG.HT.Pr_s')[0], 3), ' | Pr_t: ', round(printObj.get_val('top_group.CG.HT.Pr_t')[0], 3))
    print('\n Nu_D_s: ', round(printObj.get_val('top_group.CG.HT.Nu_D_s')[0], 3), ' | Nu_D_t: ', round(printObj.get_val('top_group.CG.HT.Nu_D_t')[0], 3))

    print('\n NTU Effectiveness: ', round(printObj.get_val('top_group.CG.NTU_effectiveness')[0], 3))
    print('\n Tube-side mass flow rate (kg/s)', round(printObj.get_val('top_group.CG.m_dot_t')[0], 3), ' | Shell-side mass flow rate (kg/s)', round(printObj.get_val('top_group.CG.m_dot_s')[0], 3))
    print('\n Tube-side outlet temperature (C): ', round(constants.convert_temperature(printObj.get_val('top_group.CG.T_out_t')[0], 'K', 'C'), 3), ' | Shell-side outlet temperature (C): ', round(constants.convert_temperature(printObj.get_val('top_group.CG.T_out_s')[0], 'K', 'C')))
    print('\n Tube-side temperature delta (C): ', round(printObj.get_val('top_group.CG.dT_t')[0], 3))

    print('\n Radiator core height (m): ', round(printObj.get_val('top_group.CG.H_core')[0], 3), ' | Radiator core depth (m): ', round(printObj.get_val('top_group.CG.D_core')[0], 3), ' | Radiator core length (m): ', round(printObj.get_val('top_group.CG.HT.L_tube')[0], 3))
    print('\n Number of tubes perpendicular to flow: ', round(printObj.get_val('top_group.CG.NT')[0], 3), ' | Number of tubes parallel to flow: ', round(printObj.get_val('top_group.CG.NL')[0], 3))
    print('\n ST/D: ', round(printObj.get_val('top_group.CG.ST_D')[0], 3), ' | SL/D: ', round(printObj.get_val('top_group.CG.SL_D')[0], 3))
    print('\n Tube outer diameter (mm): ', round(printObj.get_val('top_group.CG.HT.OD_tube')[0]*1000, 3), ' | Tube inner diameter (mm): ', round(printObj.get_val('top_group.CG.HT.ID_tube')[0]*1000, 3), ' | Tube wall thickness (mm): ', round(printObj.get_val('top_group.CG.HT.wall_tube_output')[0]*1000, 4))
    print('\n Total (wet) mass of core (kg): ', round(printObj.get_val('top_group.CG.core_wet_mass')[0], 3), ' | Fluid mass in core (kg): ', round(printObj.get_val('top_group.CG.fluid_mass')[0], 3), ' | Dry mass of core (kg): ', round(printObj.get_val('top_group.CG.dry_mass')[0], 3))

    print('\n Tube-side pressure loss: ', round(printObj.get_val('top_group.PL.dP_maj_t')[0], 3), 'Pa (', round(printObj.get_val('top_group.PL.dP_maj_t')[0]/6895, 3), 'psid )')
    print('\n Shell-side major pressure loss: ', round(printObj.get_val('top_group.CG.dP_maj_s')[0], 3), 'Pa')

    print('\n Darcy friction factor: ', round(printObj.get_val('top_group.DFF.f')[0], 3), ' | Tube-side Reynolds number: ', round(printObj.get_val('top_group.CG.HT.Re_D_t')[0], 3))

class inputConstructor():
    def __init__(self):

        #General user inputs
        self.shell_inlet_temp = None #C
        self.tube_inlet_temp = None #C
        self.OD_tube = None #m
        self.wall_tube = None #m
        self.heat_transfer = None #W
        self.U_penalty = None #penalty on overall heat transfer coefficient U from 0 to 1, 0 = no penalty, 1 = no heat transfer
        self.staggered = None #True or false for staggered tube arrangement

        #Environment parameters
        self.shell_ref_pressure = None #Pa
        self.tube_ref_pressure = None #Pa
        self.relative_humidity = None # from 0 to 100
        self.EGW_percentage = None # Ethylene glycol content in water from 0 to 100
        self.tube_absolute_roughness = None; #m, absolute roughness of inner surface of tubes
        self.rho_tube = None #kg/m^3, density of tube material
        self.k_tube = None #W/mK, conductivity of tube material; ~25 for steel

        #Contraint Variables - Defaults
        self.dT_t_max = None #C
        self.core_height_max = None #m
        self.core_length_max = None #m
        self.core_depth_max = None #m
        self.min_effectiveness = None
        self.max_dP_t_psi = None #psi
        self.max_dP_s_Pa = None #Pa
        self.min_NT = None
        self.min_NL = None #must be 10 for valid model
        self.max_NT = None
        self.max_NL = None
        self.min_ST_D = None
        self.min_SL_D = None
        self.max_ST_D = None
        self.max_SL_D = None

        #eval variables
        self.T_out_ts = None
        self.T_out_ss = None
        self.SL_D = None
        self.ST_D = None
        self.NL = None
        self.NT = None
        self.L_tube = None

def getCSV(prob, filename):
    final_inputs = prob.model.list_inputs(units=True, prom_name=True, out_stream=None, return_format='dict')
    final_outputs = prob.model.list_outputs(units=True, prom_name=True, out_stream=None, return_format='dict')

    # Extract variable names and values
    data_to_write = {**final_inputs, **final_outputs}
    var_names = list(data_to_write.keys())
    var_values = [data_to_write[name]['val'][0] for name in var_names]

    # Write the data to a CSV file.
    csv_filename = filename + '.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(var_names)

        # Write the data row
        writer.writerow(var_values)

    print(f'Final problem data saved to "{csv_filename}"')