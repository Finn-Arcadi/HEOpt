#Pre-requisites:
#openmdao, pyfluids, ht, numpy

import numpy as np
import openmdao.api as om
import ht as ht
from pyfluids import Fluid, FluidsList, Input, HumidAir, InputHumidAir

#The first class will be for the tube-side fluid calculations

class HeatTransfer(om.ExplicitComponent):
        
    def setup(self):
        #--------Tube-Side
        #User inputs
        self.add_input('Qdot', units='W', desc='Target heat transfer rate')
        self.add_input('T_in_t', units='K', desc='Hot-side temperature of the fluid')
        self.add_input('staggered', desc='Boolean for staggering of core tubes')
        self.add_input('EGW_percentage', desc='Percentage of ethylene glycol in tube-side water')

        #Design variables
        self.add_input('T_out_t', units='K', desc='Hot-side outlet temperature of the fluid')

        #Other
        self.add_input('Darcy_f', desc='Darcy friction factor for tube pressure loss')

        #Heat transfer coefficient outputs
        self.add_output('rho_t', units='kg/m**3', desc='Tube-side fluid density')
        self.add_output('mu_t', units='Pa*s', desc='Tube-side fluid dynamic viscosity')
        self.add_output('k_t', units='W/(m*K)', desc='Tube-side fluid thermal conductivity')
        self.add_output('cP_t', units='J/(kg*K)', desc='Tube-side fluid specific heat capacity at constant pressure')
        self.add_output('Re_D_t', desc='Tube-side Reynolds number')
        self.add_output('Pr_t', desc='Tube-side Prandtl number')
        self.add_output('Nu_D_s', desc='Shell-side Nusselt number')
        self.add_output('Nu_D_t', desc='Tube-side Nusselt number')

        #--------Shell-Side
        #User inputs
        self.add_input('T_in_s', units='K', desc='Cold-side temperature of the fluid')

        #Design variables
        self.add_input('T_out_s', units='K', desc='Cold-side outlet temperature of the fluid')

        #Heat transfer coefficient outputs
        self.add_output('rho_s', units='kg/m**3', desc='Shell-side fluid density')
        self.add_output('mu_s', units='Pa*s', desc='Shell-side fluid dynamic viscosity')
        self.add_output('k_s', units='W/(m*K)', desc='Shell-side fluid thermal conductivity')
        self.add_output('cP_s', units='J/(kg*K)', desc='Shell-side fluid specific heat capacity at constant pressure')
        self.add_output('Re_D_s', desc='Shell-side Reynolds number')
        self.add_output('Pr_s', desc='Shell-side Prandtl number')

        #--------Tubes
        #User inputs
        self.add_input('OD_tube', units='m', desc='Tube outer diameter')
        self.add_input('wall_tube', units='m', desc='Wall thickness of tube')
        self.add_input('k_tube', units='W/(m*K)', desc='Thermal conductivity of tube material')
        self.add_input('rho_tube', units='kg/m**3', desc='Density of tube material')
        self.add_input('U_penalty', desc = 'Overall heat transfer coefficient penalty, dimensionless')
        self.add_input('Shell_Ref_Pressure', desc='Reference pressure for shell-side fluid properties')
        self.add_input('Tube_Ref_Pressure', desc='Reference pressure for tube-side fluid properties')
        self.add_input('Relative_Humidity', desc='Relative humidity of air on shell-side fluid')
        
        #Design variables
        self.add_input('ST_D', desc='Tube spacing geometry factor perpendicular to flow direction')
        self.add_input('SL_D', desc='Tube spacing geometry factor in flow direction')
        self.add_input('NT', desc='Number of tubes in perpendicular to flow direction')
        self.add_input('NL', desc='Number of tube layers in flow direction')

        #Cycle inputs
        self.add_input('L_tube', units='m', desc='Length of tubes')

        #Coupling outputs
            #Core geometry
        self.add_output('A_req', units='m**2', desc='Required heat transfer area')
        self.add_output('N_tubes', desc='Number of tubes in radiator')
            #Tube-side pressure loss
        self.add_output('V_t', units='m/s', desc='Tube-side flow velocity')

        #Other outputs
        self.add_output('Q_dot_NTU', units='W', desc='Heat transfer rate calculated using NTU method')
        self.add_output('Q_dot_LMTD', units='W', desc='Heat transfer rate calculated using LMTD method')
        self.add_output('design_effectiveness', desc='Effectiveness based on design conditions')
        self.add_output('NTU_effectiveness', desc='Effectiveness based on NTU method')
        self.add_output('D_core', units='m', desc='Depth of radiator core')
        self.add_output('H_core', units='m', desc='Height of radiator core')
        self.add_output('m_dot_t', units='kg/s', desc='Mass flow rate of hot-side fluid')
        self.add_output('m_dot_s', units='kg/s', desc='Mass flow rate of cold-side fluid')
        self.add_output('fluid_mass', units='kg', desc='Fluid mass inside  tubes')
        self.add_output('dry_mass', units='kg', desc='Dry tube mass')
        self.add_output('core_wet_mass', units='kg', desc='Wet mass of core')
        self.add_output('a', desc='Dimensionless tube spacing SL/D in flow direction')
        self.add_output('b', desc='Dimensionless tube spacing ST/D perpendicular to flow direction')
        self.add_output('dT_t', units='K', desc='Hot-side inlet to outlet temperature difference')
        self.add_output('ID_tube', units='m', desc='Pass-through so that other classes may use the tube ID')
        self.add_output('dP_maj_s', units='Pa', desc='Shell side major pressure loss')
        self.add_output('wall_tube_output', units = 'm', desc = 'Tube wall thickness')

        #Heat transfer coefficient outputs
        self.add_output('U', units='W/((m**2)*K)', desc='Overall heat transfer coefficient')
        self.add_output('h_s', units='W/((m**2)*K)', desc='Shell-side convective heat transfer coefficient')
        self.add_output('h_t', units='W/((m**2)*K)', desc='Tube-side convective heat transfer coefficient')
        self.add_output('R_tube', units='((m**2)*K)/W', desc='Tube wall thermal resistance')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation heat transfer between the fluids
        """

        #Inputs
        #General inputs
        Qdot = inputs['Qdot']
        staggered = inputs['staggered']

        #Tube-side inputs
        T_in_t = inputs['T_in_t']
        T_out_t = inputs['T_out_t']
        Tube_ref_p = inputs['Tube_Ref_Pressure']
        Percent_EGW = inputs['EGW_percentage']

        #Shell-side inputs
        T_in_s = inputs['T_in_s']
        T_out_s = inputs['T_out_s']

        #Shell-side properties
        Shell_ref_p = inputs['Shell_Ref_Pressure']
        Shell_RH = inputs['Relative_Humidity']

        #Tube inputs
        OD_tube = inputs['OD_tube']
        wall_tube = inputs['wall_tube']
        k_tube = inputs['k_tube']
        rho_tube = inputs['rho_tube']
        U_penalty = inputs['U_penalty']
        L_tube = inputs['L_tube']

        ST_D = inputs['ST_D']
        SL_D = inputs['SL_D']

        ST = ST_D*OD_tube
        SL = SL_D*OD_tube

        ID_tube = OD_tube - 2*wall_tube
        
        NT = inputs['NT']
        NL = inputs['NL']

        #Calculate average fluid properties
        T_avg_s = (T_in_s + T_out_s)/2
        T_avg_t = (T_in_t + T_out_t)/2

        ethylene_glycol = Fluid(FluidsList.MEG, np.clip(Percent_EGW, a_min=0, a_max=None)).with_state(
            Input.pressure(Tube_ref_p), Input.temperature(T_avg_t - 273.15))

        air = HumidAir().with_state(
            InputHumidAir.pressure(Shell_ref_p),
            InputHumidAir.temperature(T_avg_s - 273.15),
            InputHumidAir.relative_humidity(Shell_RH),
        )

        cP_s = air.specific_heat
        rho_s = air.density
        mu_s = air.dynamic_viscosity
        k_s = air.conductivity
        cP_t = ethylene_glycol.specific_heat
        rho_t = ethylene_glycol.density
        mu_t = ethylene_glycol.dynamic_viscosity
        k_t = ethylene_glycol.conductivity

        #Calculate mass flow rate
        m_dot_t = Qdot / (cP_t * (T_in_t - T_out_t))
        m_dot_s = Qdot / (cP_s * (T_out_s - T_in_s))

        #Calculate the log mean temperature difference
        delta_T_max = T_in_t - T_out_s
        delta_T_min = T_out_t - T_in_s

        if delta_T_min == delta_T_max: #If LMTD values are the same, we assume the mean temperature is the dT
            LMTD = delta_T_min
        else:
            LMTD = (delta_T_max - delta_T_min) / np.log(delta_T_max / delta_T_min)

        #Calculate the effectiveness
        design_effectiveness = (m_dot_t*cP_t*(T_in_t - T_out_t))/(m_dot_s*cP_s*(T_in_t - T_in_s))

        #Calculate radiator core geometry
        N_tubes = NT * NL #number of tubes

        H_core = ST * NT #height of the radiator core
        D_core = SL * NL #depth of the radiator core

        # a = SL/OD_tube #longitudinal pitch ratio
        # b = ST/OD_tube #transverse pitch ratio

        #Calculate the tube thermal resistance
        R_tube = np.log(OD_tube/ID_tube) / (2*np.pi*k_tube*L_tube)

        #Calculate the cold-side convective heat transfer coefficient
        L_char = OD_tube #(4*L_tube*(ST - OD_tube))/(2*(ST - OD_tube) + 2*L_tube) #characteristic length for cold-side
        A_flow_s = NT*L_tube*(ST-OD_tube) #flow area for cold-side
        V_dot_s = m_dot_s / rho_s #volumetric flow rate for cold-side
        V_s = V_dot_s / A_flow_s #velocity for cold-side
        Re_D_s = (rho_s * V_s * L_char) / mu_s #Reynolds number for cold-side  

        Pr_s = (cP_s * mu_s) / k_s #Prandtl number for cold-side
        
        Nu_D_s_pure = Zukauskas_Nu(Re=Re_D_s, Pr=Pr_s, b=ST_D, a=SL_D, staggered = staggered)

        Nu_correction = ht.Zukauskas_tube_row_correction(NL, staggered=staggered, Re=Re_D_s)

        Nu_D_s = Nu_D_s_pure * Nu_correction

        h_s = (Nu_D_s * k_s) / L_char #convective heat transfer coefficient for cold-side

        #Calculate the hot-side convective heat transfer coefficient
        A_flow_t = N_tubes * (np.pi * (ID_tube**2) / 4) #flow area for hot-side
        V_dot_t = m_dot_t / rho_t #volumetric flow rate for hot-side
        V_t = V_dot_t / A_flow_t #velocity for hot-side
        Re_D_t = (rho_t * V_t * ID_tube) / mu_t #Reynolds number for hot-side
        Pr_t = (cP_t * mu_t) / k_t #Prandtl number for hot-side
        if Re_D_t > 2300:
            if T_in_s < T_in_t: #Cooling the hot fluid
                Nu_D_t = 0.023 * (Re_D_t**0.8) * (Pr_t**0.3) #Nusselt number for hot-side
            else: #Heating the "hot" fluid (i.e. the fluid in the tube)
                Nu_D_t = 0.023 * (Re_D_t**0.8) * (Pr_t**0.4) #Nusselt number for hot-side
        else:
            Nu_D_t = 4.36 #Assumes constant heat flux
        
        h_t = (Nu_D_t * k_t) / ID_tube #convective heat transfer coefficient for hot-side

        #Calculate overall heat transfer coefficient
        U = (1 - U_penalty) * (1 / ((1/h_t) + (R_tube * np.pi * OD_tube * L_tube) + (1/h_s))) #overall heat transfer coefficient

        #Calculate required heat transfer area
        A_req = Qdot / (U * LMTD) #required heat transfer area

        #Check the heat transfer and effectiveness with NTU method
        C_min = min(m_dot_t*cP_t, m_dot_s*cP_s)
        C_max = max(m_dot_t*cP_t, m_dot_s*cP_s)
        C_r = C_min / C_max
        NTU = (U * A_req) / C_min

        NTU_effectiveness = 1 - np.exp((1/C_r) * (NTU**0.22) * (np.exp(-C_r * (NTU**0.78)) - 1))
        Q_dot_NTU = NTU_effectiveness * C_min * (T_in_t - T_in_s)
        Q_dot_LMTD = U * A_req * LMTD #sanity check

        #Estimate the mass of the radiator

        tube_fluid_volume = (np.pi/4) * (ID_tube**2) * L_tube * N_tubes #volume of fluid inside tubes
        fluid_mass = tube_fluid_volume * rho_t #mass of fluid inside tubes

        tube_volume = (np.pi/4) * (OD_tube**2) * L_tube * N_tubes  - tube_fluid_volume #volume of tube material
        dry_mass = tube_volume * rho_tube #mass of tubes

        #Calculate shell-side major pressure loss
        if staggered == 1:
            #Staggered:
            psi = Re_D_s**(-0.16*(1 + 0.47 / ((SL_D - 1)**(1.08))))
        else:           
            #Inline:
            psi = Re_D_s**(-0.15*(0.176 + 0.32*ST_D / ((SL_D - 1)**(0.43 + 1.13/ST_D))))


        dP_maj_s = 0.5 * NL * psi * rho_s * (V_s**2) #Pa

        #OUTPUTS SECTION

        #Coupling outputs
            #Core geometry
        outputs['A_req'] = A_req
        outputs['N_tubes'] = N_tubes
        
        #Other Outputs
        outputs['Q_dot_NTU'] = Q_dot_NTU
        outputs['Q_dot_LMTD'] = Q_dot_LMTD
        outputs['design_effectiveness'] = design_effectiveness
        outputs['NTU_effectiveness'] = NTU_effectiveness
        outputs['m_dot_t'] = m_dot_t
        outputs['m_dot_s'] = m_dot_s
        outputs['H_core'] = H_core
        outputs['D_core'] = D_core
        outputs['fluid_mass'] = fluid_mass
        outputs['dry_mass'] = dry_mass
        outputs['core_wet_mass'] = fluid_mass + dry_mass
        outputs['dT_t'] = T_in_t - T_out_t
        outputs['V_t'] = V_t
        outputs['ID_tube'] = ID_tube
        outputs['wall_tube_output'] = wall_tube
        outputs['dP_maj_s'] = dP_maj_s

        #Heat transfer outputs
        outputs['U'] = U
        outputs['h_t'] = h_t
        outputs['h_s'] = h_s
        outputs['R_tube'] = R_tube

        #Property outputs
        outputs['rho_s'] = rho_s
        outputs['mu_s'] = mu_s
        outputs['k_s'] = k_s
        outputs['cP_t'] = cP_s

        outputs['rho_t'] = rho_t
        outputs['mu_t'] = mu_t
        outputs['k_t'] = k_t
        outputs['cP_t'] = cP_t

        outputs['Re_D_s'] = Re_D_s
        outputs['Re_D_t'] = Re_D_t
        outputs['Pr_s'] = Pr_s
        outputs['Pr_t'] = Pr_t
        outputs['Nu_D_s'] = Nu_D_s
        outputs['Nu_D_t'] = Nu_D_t


class SolveArea(om.ExplicitComponent):
    def setup(self):
        #--------Tubes
        
        #User inputs
        self.add_input('OD_tube', units='m', desc='Tube outer diameter')

        #Coupling inputs
        self.add_input('N_tubes', desc='Number of tubes in radiator')
        self.add_input('A_req', units='m**2', desc='Required heat transfer area')

        #Coupling outputs
        self.add_output('L_tube_req', units='m', desc='Length of tubes')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the required length to meet the required area
        """
        #Solve A_req = N_tubes * pi * OD_tube * L_tube for L_tube
        OD_tube = inputs['OD_tube']
        N_tubes = inputs['N_tubes']
        A_req = inputs['A_req']

        L_tube_req = A_req / (N_tubes * np.pi * OD_tube)

        outputs['L_tube_req'] = L_tube_req

class DarcyFrictionFactorComp(om.ImplicitComponent):
    """
    Evaluates the Darcy friction factor via the Colebrook-White equation to estimate tube-side pressure loss
    """
    def setup(self):
        self.add_input('epsilon', units='m', desc='Absolute surface roughness')
        self.add_input('ID_tube', units='m', desc='Tube inner diameter')
        self.add_input('Re_D_t',  desc='Tube-side Reynolds number')

        self.add_output('f', desc='Darcy friction factor', lower=1e-8)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        epsilon = inputs['epsilon']
        ID_tube = inputs['ID_tube']
        Re_D_t = inputs['Re_D_t']
        f = outputs['f']

        if Re_D_t > 2300:
            residuals['f'] = (1 / (-2 * np.log10(epsilon/(3.7 * ID_tube) + 2.51/(Re_D_t * np.sqrt(f)))))**2 - f #Colebrook-White equation
        else:
            residuals['f'] = 64/Re_D_t - f

    def guess_nonlinear(self, inputs, outputs, resids):
        Re_D_t = inputs['Re_D_t']
        # It is better to do this only if residuals are large, but in this case we will spend the extra computing power
        # as there were some issues when doing this
        outputs['f'] = 64/Re_D_t

class PressureLoss(om.ExplicitComponent):
    def setup(self):
        #Inputs
        self.add_input('Darcy_f', desc='Darcy friction factor for tube pressure loss')
        self.add_input('Length', units='m', desc='Length of tubes')
        self.add_input('ID_tube', units='m', desc='Tube inner diameter')
        self.add_input('rho_t', units='kg/m**3', desc='Density of hot-side fluid')
        self.add_input('V_t', units='m/s', desc='Velocity of tube-side fluid')

        #Outputs
        self.add_output('dP_maj_t', val=0, units="Pa", desc='Tube-side pressure drop')

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        #Inputs
        Darcy_f = inputs['Darcy_f']
        Length = inputs['Length']
        ID_tube = inputs['ID_tube']
        rho_t = inputs['rho_t']
        V_t = inputs['V_t']

        #Estimate tube-side pressure loss
        dP_maj_t = Darcy_f * (Length / ID_tube) * (rho_t * (V_t**2) / 2)

        #Tube-side
        outputs['dP_maj_t'] = dP_maj_t

class CycleGeometry(om.Group):
    """
    Group containing the heat transfer and geometry components
    to model a CycleGeometry.
    """

    def setup(self):
        #cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        self.add_subsystem('HT', HeatTransfer(), promotes_inputs=['Qdot', 'T_in_t', 'T_out_t',
                        'T_in_s', 'T_out_s',
                        'OD_tube', 'wall_tube', 'k_tube', 'rho_tube',
                        'U_penalty', 'ST_D', 'SL_D', 'NT', 'NL'], 
                            promotes_outputs=['fluid_mass', 'dry_mass', 'core_wet_mass',
                                            'Q_dot_NTU', 'Q_dot_LMTD', 'design_effectiveness',
                                            'NTU_effectiveness', 'D_core', 'H_core',
                                            'm_dot_t', 'm_dot_s',
                                            'U', 'h_t', 'h_s', 'R_tube',
                                            'a', 'b', 'dT_t', 'dP_maj_s'])
        
        self.add_subsystem('SA', SolveArea(), promotes_inputs=['OD_tube'])

        # Connect the remaining parameters between geometry solver and heat transfer solver
        self.connect('SA.L_tube_req', 'HT.L_tube')
        self.connect('HT.N_tubes', 'SA.N_tubes')
        self.connect('HT.A_req', 'SA.A_req')
        #by promotion, N_tubes and A_req are also connected

        # Nonlinear Block Gauss Seidel is a gradient free solver

        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['maxiter'] = 1000

        #self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        #self.nonlinear_solver.options['maxiter'] = 10
        #self.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='wall')
        #self.nonlinear_solver.options['linesearch'] = 'BoundsEnforceLS'
        #self.linear_solver = om.DirectSolver()
        #self.linear_solver.options['rhs_checking'] = True

def Zukauskas_Nu(Re, Pr, b, a, staggered, Pr_wall=None):
    """
    Function returns the Nusselt number for staggered or in-line tube banks. Primarily based on
    Zukauskas' 1972 article titled "Heat Transfer from Tubes in Crossflow", with some flow regimes filled
    in from Pypi's HT library.

    Pr_wall input is optional, and provides a correction factor.

    m=0.5 and c=0.73 for single tube at Re > 40 noted by Zukauskas, and suggested for use in the
    transitional in-line regime (100 < Re < 1000)
    """

    f = 1.0
    if staggered == 0: #inline case
        if Re < 100:
            c, m = 0.8, 0.4 #eq 38
        elif Re < 1000:
            c, m = 0.73, 0.5 #single tube values as suggested by Zukauskas
        elif Re < 2E5:
            c, m = 0.27, 0.63 #eq 39
        else:
            c, m = 0.021, 0.84
    else: #staggered case
        if Re < 100:
            c, m = 0.9, 0.4 #eq 38
        elif Re < 1000:
            c, m = 0.73, 0.5 #single tube values as suggested by Zukauskas
        elif Re < 2E5:
            m = 0.6 #eq 40 or 41
            if a/b < 2:
                c = 0.35 #eq 40
                f = (a/b)**0.2 #eq 40
            else:
                f = 1 #eq 41
                c = 0.4 #eq 41
        else:
            if(Pr > 1):
                c, m = 0.022, 0.84 #eq 43
            else:
                c, m = 0.019, 0.84 #eq 44, for Pr=0.7

    Nu = c * (Re**m) * (Pr**0.36) * f

    if Pr_wall is not None:
        Nu*= (Pr/Pr_wall)**0.25

    return Nu