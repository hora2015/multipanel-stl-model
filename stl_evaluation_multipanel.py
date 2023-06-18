# ## Sound Transmission Loss (STL) Evaluation for Multipanel System

# #### ---------------------------------------------------------------------------------------------------
#
# ### Libraries and Constants
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import acoustics as ac
import cmath
from scipy.optimize import fsolve, ridder
import random

# set constants
_c_air = 343.3  #m/s at room temp of 20°C
_rho_air = 1.204  #kg/m^3 at room temp of 20°C
_panel_height = 3  #in m (ISO 354)
_panel_width = 4  #in m (ISO 354)
_max_panels = 3 #limited to triple panel system


# #### ---------------------------------------------------------------------------------------------------

# ### Classes


class AcousticPanel:
    
    """
    The AcousticPanel class calculates the transmission loss values of an acoustic panel given certain properties.
    
    Attributes:
    height (float): the height of the acoustic panel in meters
    width (float): the width of the acoustic panel in meters
    thickness (float): the thickness of the acoustic panel in meters
    elastic_modulus (float): the elastic modulus of the material in the acoustic panel in pascals
    density (float): the density of the material in the acoustic panel in kilograms per cubic meter
    poisson_ratio (float): the Poisson's ratio of the material in the acoustic panel
    damping_factor (float): the damping factor of the material in the acoustic panel
    
    Methods:
    transmission_loss(min_freq, max_freq): calculates the transmission loss values of the acoustic panel 
    over a range of frequencies
    """
    
    def __init__(self, thickness, elastic_modulus, density, poisson_ratio, damping_factor, height, width):
        """
        Initializes an AcousticPanel object with the given properties.

        Args:
        - thickness (float): the thickness of the acoustic panel in meters
        - elastic_modulus (float): the elastic modulus of the material in the acoustic panel in pascals
        - density (float): the density of the material in the acoustic panel in kilograms per cubic meter
        - poisson_ratio (float): the Poisson's ratio of the material in the acoustic panel
        - damping_factor (float): the damping factor of the material in the acoustic panel
        - height (float): the height of the acoustic panel in meters
        - width (float): the width of the acoustic panel in meters
        """
        self.height = height
        self.width = width
        self.thickness = thickness
        self.elastic_modulus = elastic_modulus
        self.density = density
        self.poisson_ratio = poisson_ratio
        self.damping_factor = damping_factor
        
        self.basis_wt = self.density * self.thickness
        self.flexural_rigidity = self.elastic_modulus*self.thickness**3/(12 * (1-self.poisson_ratio**2))
        self.shear_modulus = self.elastic_modulus/(2*(1+self.poisson_ratio))
        
#         print(self.basis_wt)
#         print(self.flexural_rigidity)
        
    def _calc_critical_frequency(self):
        """
        Calculates the critical frequency of the acoustic panel.

        Returns:
        - critical_frequency (float): the critical frequency of the acoustic panel
        """
        self.critical_frequency = (_c_air**2)/(2*np.pi) * np.sqrt(self.basis_wt/self.flexural_rigidity)
        return self.critical_frequency
    
    def _calc_fundamental_frequency(self):
        """
        Calculates the fundamental frequency of the acoustic panel.

        Returns:
        - fundamental_frequency (float): the fundamental frequency of the acoustic panel
        """
        area_harmonic_mean = 1/self.width**2 + 1/self.height**2
        self.fundamental_frequency = (np.pi/2) * np.sqrt(self.flexural_rigidity/self.basis_wt)
        self.fundamental_frequency *= area_harmonic_mean
        return self.fundamental_frequency
    
    def _calc_shear_frequency(self):
        """
        Calculates the shear modulus of the material in the acoustic panel.

        Returns:
        - shear_frequency (float): the shear frequency of the material in the acoustic panel
        """
        self._calc_critical_frequency()
        self.shear_frequency = (_c_air**2)*(1-self.poisson_ratio)
        self.shear_frequency /= (59 * self.thickness**2 * self.critical_frequency)
        return self.shear_frequency
    

    def stl_at_frequency(self, freq):
        """
        Calculates transmission loss at a given frequency 
        
        Parameters:
        - freq (float): frequency at which sound transmission loss is calculated

        Returns:
        - tl (float): transmission loss value in dB
        """
        
        self._calc_critical_frequency()
        self._calc_fundamental_frequency()
        self._calc_shear_frequency()
#         print('fp = ' + str(self.fundamental_frequency))
#         print('fc = ' + str(self.critical_frequency))
#         print('fs = ' + str(self.shear_frequency))
        
        if self.shear_frequency < self.critical_frequency:
            shear_dominates = True
#             print('shear dominates')
        else:
            shear_dominates = False
        
        w = 2 * np.pi * freq
            
        """ Calculate mass-law transmission loss """
        tl_mass = 10 * np.log10(1 + (w * self.basis_wt/(3.6 * _c_air * _rho_air))**2)   #fp < freq < fc

        tl = tl_mass

        if freq <= self.fundamental_frequency:
            tl = tl_mass + 40 * np.log10(self.fundamental_frequency / freq)

        if shear_dominates:
            if freq >= self.shear_frequency:
                tl = tl_mass - 6
#                     print('shear dominates')
        elif freq >= self.critical_frequency:
            if freq == self.critical_frequency:
                delta_f = ac.octave.upper_frequency(frequency=freq, fraction=3) - \
                ac.octave.lower_frequency(frequency=freq, fraction=3)
            else:
                delta_f = freq - self.critical_frequency
            tl = 20 * np.log10(w*self.basis_wt/(2 * _c_air * _rho_air)) +  \
                       10 * np.log10(2*self.damping_factor/np.pi * delta_f/self.critical_frequency)

            if freq >= self.shear_frequency: #mass law minus 6dB
                tl = tl_mass - 6
#                     print(freq, tl)
          
        return tl
    
    
    def transmission_loss_range(self, frequency_range):
        """
        Calculates the transmission loss values of the acoustic panel over a range of frequencies.
        
        Args:
        - frequency_range (list): a list containing the minimum and maximum frequencies of the range in Hertz

        Returns:
        - tl_df (DataFrame): A dataframe with 1/3rd octave frequencies in Hz and transmission loss values in dB.
        """

        """ Calculates a list of 1/3 octave frequencies between the given minimum and maximum frequencies. """
        if (len(frequency_range) > 2) or (len(frequency_range) < 1):
            min_freq = 20
            max_freq = 8000
        elif len(frequency_range) == 2:
            min_freq = frequency_range[0]
            max_freq = frequency_range[1]
        
        freq_bands = list(ac.bands.third(min_freq, max_freq))
        
        tl_df = pd.DataFrame()
        
        for i, freq in enumerate(freq_bands):
            tl = self.stl_at_frequency(freq)
            df = pd.DataFrame(data={'Frequency': [float(freq)], 'Transmission_Loss': [tl]}, index=[i])
            tl_df = pd.concat([tl_df, df])
        
        return tl_df

class AbsorberPanel:
    """
    A class to represent an absorber panel used in acoustic engineering.

    Attributes:
    -----------
    thickness : float
        The thickness of the absorber panel in meters.
    density : float
        The density of the absorber panel material in kg/m^3.
    air_flow_resistivity : float
        The air flow resistivity of the absorber panel material in Ns/m^4.
    porosity : float
        The porosity of the absorber panel material.
    tortuosity : float
        The tortuosity of the absorber panel material.
    bulk_modulus : float
        The bulk modulus of the absorber panel material in Pa.

    Methods:
    --------
    miki_propagation_constant(frequency)
        Calculates the Miki propagation constant of the absorber panel at a given frequency.
    transmission_loss(frequency_range)
        Calculates the transmission loss of the absorber panel over a given frequency range.
    
    """

    
    def __init__(self, category, thickness, density, air_flow_resistivity, solid_density, 
                 solid_elastic_modulus, solid_poisson_ratio, damping_factor):
        """
        Initializes an AbsorberPanel object with the given properties.

        Args:
        - category (str): The category of the absorber panel.
        - thickness (float): The thickness of the absorber panel in meters.
        - density (float): The density of the absorber panel material in kg/m^3.
        - air_flow_resistivity (float): The air flow resistivity of the absorber panel material in Ns/m^4.
        - solid_density (float): The density of the solid material in the absorber panel in kg/m^3.
        - solid_elastic_modulus (float): The elastic modulus of the solid material in the absorber panel in Pa.
        - solid_poisson_ratio (float): The Poisson's ratio of the solid material in the absorber panel.
        - damping_factor (float): The damping factor of the absorber panel.

        """
        self.category = category
        self.thickness = thickness
        self.density = density
        self.air_flow_resistivity = air_flow_resistivity
        self.solid_density = solid_density
        self.porosity = 1 - (self.density/self.solid_density)
        self.solid_elastic_modulus = solid_elastic_modulus
        self.solid_poisson_ratio = solid_poisson_ratio
        self.damping_factor = damping_factor
        
        self.basis_wt = self.density * self.thickness
        
        
    def tortuosity(self):
        """
        Calculates the tortuosity of the absorber panel based on the 2012 Matyka and Ahmadi models.

        Returns:
        --------
        tau (float): The tortuosity of the absorber panel.

        """
        
        if self.category == "Air":
            tau = 1
        else:
            B_compact = 1.09  #2012 Matyka model
            phi = self.porosity
            tau = np.sqrt(1/3 + 2/3 * phi/(1 - B_compact*(1-phi)**(2/3)) )
        
        return tau

    
    def _porous_poisson(self, *parameters):
        """
        Helper function for calculating the elastic modulus of the absorber panel material.
        It solves the equation for porous Poisson's ratio.

        Args:
        - parameters (tuple): A tuple containing the parameters required for the equation.



        Returns:
        --------
        func (float): The result of the equation for porous Poisson's ratio.

        """

        v_c, porosity, v_m = parameters
        _limit_v = 1/5
        
        term1 = (1 + v_m) / (1 + v_c)
        term2 = (1 - v_m) / (1 - v_c)
        term3 = (v_c - _limit_v) / (v_m - _limit_v)

        func = 1 - porosity - (term1)**(2/3) * (term2)**(1/6) * (term3)**(5/6)
            
        return func
    
    
    def elastic_modulus(self):
        """
        Calculates the elastic modulus and Poisson's ratio of the absorber panel material.

        Returns:
        --------
        E_c (float): The elastic modulus of the absorber panel material.
        v_c (float): The Poisson's ratio of the absorber panel material.

        """
        _limit_v = 1/5
        _eps = 1e-6

        phi = self.porosity
        v_m = self.solid_poisson_ratio
        parameters = (phi, v_m)
        low_vc = _limit_v + _eps
        high_vc = self.solid_poisson_ratio
        v_c = ridder(self._porous_poisson, a=low_vc, b=high_vc, args=parameters)

        term1 = (1 + v_m) / (1 + v_c)
        term2 = (v_c - _limit_v) / (v_m - _limit_v)
        E_c = self.solid_elastic_modulus * (term1)**(2/3) * (term2)**(5/3)
            
        return E_c, v_c
    
    
    def bulk_modulus(self):
        """
        Calculates the bulk modulus of the absorber panel material.

        Returns:
        --------
        K_c (float): The bulk modulus of the absorber panel material.

        """

        if self.category == "Air":
            K_c = _rho_air * (_c_air ** 2)
        else:
            E_c, v_c = self.elastic_modulus()
            K_c = E_c/(3 * (1 - 2*v_c))
        
        return K_c
    
    
    def miki_propagation_coefficient(self, frequency):
        """
        Calculates the 3-parameter Miki propagation constant, gamma(f)
        gamma(f) = alpha(f) + j * beta(f)
        alpha : attenuation constant
        beta : phase constant
        
        Parameters:
        -----------
        frequency (float): The frequency at which to calculate the Miki propagation constant.

        Returns:
        --------
        gamma (complex): The Miki propagation constant for the absorber panel at the given frequency.

        """

        omega = 2 * np.pi * frequency
        effective_resistivity = self.porosity * self.air_flow_resistivity/(self.tortuosity()**2)
        
        sp_wave_num = omega * self.tortuosity() / _c_air
        sp_freq = frequency / effective_resistivity
        exponent = -0.618
        freq_term = sp_freq ** exponent 
        
        alpha = sp_wave_num * 0.160 * freq_term
        beta = sp_wave_num * (1 + 0.109 * freq_term)
        gamma = complex(alpha, beta)
        
        return gamma
        
    def stl_at_frequency(self, freq):
        """
        Calculates the transmission loss at a given frequency using the Fahy formula.
        TL = 8.6 * alpha * d + 20 * log10(beta / k)

        Parameters:
        - freq (float): The frequency at which sound transmission loss is calculated.

        Returns:
        - tl (float): The transmission loss value in dB.
        """
        
        if self.category == "Air":
            tl = 6
        else:
            wavenumber = 2 * np.pi * freq / _c_air
            gamma = self.miki_propagation_coefficient(freq)
            alpha = gamma.real
            beta = gamma.imag
            tl = 8.6 * alpha * self.thickness + 20 * np.log10(beta / wavenumber)
        
        return tl
    
    
    def transmission_loss_range(self, frequency_range):
        """
        Calculate the transmission loss for the given frequency range using 
        Fahy formula = 8.6 * alpha * d + 20 * np.log10(beta / k), 
        which is valid above highest resonant frequency defined by c_air/(2 * pi * d)

        Parameters:
        - frequency_range (list): A list [min_freq, max_freq] specifying the minimum and maximum frequency in Hz.

        Returns:
        - tl_df: A dataframe 1/3rd octave frequencies in Hz and transmission loss values in dB.
        """
        
        
        if (len(frequency_range) > 2) or (len(frequency_range) < 1):
            min_freq = 20
            max_freq = 8000
        elif len(frequency_range) == 2:
            min_freq = frequency_range[0]
            max_freq = frequency_range[1]
        
        freq_bands = list(ac.bands.third(min_freq, max_freq))

        tl_df = pd.DataFrame()
        
        for i, freq in enumerate(freq_bands):
            tl = self.stl_at_frequency(freq)
            df = pd.DataFrame(data={'Frequency': [float(freq)], 'Transmission_Loss': [tl]}, index=[i])
            tl_df = pd.concat([tl_df, df])
        
        return tl_df


class MultiPanel:
    """
    A class representing a multi-panel structure with absorbers.

    Attributes
    ----------
    structure : dict
        A dictionary containing information about the panels and absorbers in the structure.
    n_panels : int
        The number of panels in the structure.
    n_absorbers : int
        The number of absorbers in the structure.
    panel_objs : list
        A list of `AcousticPanel` objects representing each panel in the structure.
    absorber_objs : list
        A list of `AbsorberPanel` objects representing each absorber in the structure.
    resonant_freq : float
        The resonant frequency of the structure, in Hz.
    standing_wave_freq : float
        The maximum standing wave frequency of the structure, in Hz.

    Methods
    -------
    calculate_combined_transmission_loss(freq_range)
        Calculates the transmission loss of the structure for a given frequency range.
    calculate_third_octave_frequencies(freq_range)
        Calculates the third-octave frequency bands within a given frequency range.
    """

    def __init__(self, structure_dict):
        """
        Initializes a MultiPanel object.

        Parameters
        ----------
        structure_dict : dict
            A dictionary containing information about the panels and absorbers in the structure.
        """
        
        self.structure = structure_dict
        
        self.n_panels = len(structure_dict['panels'])
        self.n_absorbers = len(structure_dict['absorbers'])
        self.panel_objs = []
        self.absorber_objs = []
        
        # create panel and absorber objects
        for panel in self.structure['panels'].items():
            panel_obj = AcousticPanel(height = _panel_height, 
                                      width = _panel_width, 
                                      thickness = panel[1]['Thickness'], 
                                      elastic_modulus = panel[1]['ElasticModulus'], 
                                      density = panel[1]['Density'], 
                                      poisson_ratio = panel[1]['PoissonRatio'], 
                                      damping_factor = panel[1]['DampingRatio']
                                     )
            self.panel_objs.append(panel_obj)
            
        for absorber in self.structure['absorbers'].items():
            absorber_obj = AbsorberPanel(category = absorber[1]['Category'],
                                         thickness = absorber[1]['Thickness'], 
                                         density = absorber[1]['Density'], 
                                         air_flow_resistivity = absorber[1]['AirFlowResistivity'], 
                                         solid_density = absorber[1]['MaterialDensity'], 
                                         solid_elastic_modulus = absorber[1]['SolidElasticModulus'], 
                                         solid_poisson_ratio = absorber[1]['SolidPoissonRatio'], 
                                         damping_factor = absorber[1]['DampingRatio']
                                        )
            self.absorber_objs.append(absorber_obj)
    
        if self.n_absorbers == self.n_panels:  #last absorber becomes another layer of last panel
            ''' basis weight of absorber adds to the last panel'''
            self.panel_objs[-1].basis_wt = self.panel_objs[-1].basis_wt + self.absorber_objs[-1].basis_wt

    def _calculate_standing_wave_frequency(self):
        """
        Calculates standing wave frequencies for the panel absorber system
        """
        
        self.standing_wave_frequency = []
        for i_gap in range(self.n_panels - 1):
            f = _c_air / (2 * np.pi * self.absorber_objs[i_gap].thickness)
            self.standing_wave_frequency.append(f)
        
        return self.standing_wave_frequency

    def _calculate_air_mass_resonance_frequency(self):
        """
        Calculates the air mass resonance frequency for the panel-absorber system.

        Returns
        -------
        float
            The air mass resonance frequency in Hz.
        """
        
        f_amr = []
        if self.n_panels == 2:
            K = self.absorber_objs[0].bulk_modulus()
            d = self.absorber_objs[0].thickness
            m1 = self.panel_objs[0].basis_wt
            m2 = self.panel_objs[1].basis_wt
            hm = 1/m1 + 1/m2
            f_alpha = 1/(2 * np.pi) * np.sqrt(K/d * hm)
            f_amr.append(f_alpha)
            
        elif self.n_panels == 3:
            d1 = self.absorber_objs[0].thickness
            d2 = self.absorber_objs[1].thickness
            
            K1 = self.absorber_objs[0].bulk_modulus()
            K2 = self.absorber_objs[1].bulk_modulus()
            
            m1 = self.panel_objs[0].basis_wt
            m2 = self.panel_objs[1].basis_wt
            m3 = self.panel_objs[2].basis_wt
            
            lam1 = K1 * m3 * (m1 + m2) / d1
            lam2 = K2 * m1 * (m2 + m3) / d2
            
            a = lam1 + lam2
            b = (lam1 - lam2)**2
            c = 4 * K1 * K2 * (m1 * m3)**2
            m123 = m1 * m2 * m3
            
            f_alpha = np.sqrt(2)/(4 * np.pi) * np.sqrt((a - np.sqrt(b + c))/m123)
            f_beta = np.sqrt(2)/(4 * np.pi) * np.sqrt((a + np.sqrt(b + c))/m123)
            f_amr.append(f_alpha)
            f_amr.append(f_beta)

        self.air_mass_resonance_frequency = f_amr
        
        return self.air_mass_resonance_frequency
    
    
    def double_panel_loss(self, freq_bands):
        
        self._calculate_air_mass_resonance_frequency()
        self._calculate_standing_wave_frequency()
        
        f_alpha = self.air_mass_resonance_frequency[0]
        f_l = self.standing_wave_frequency[0]
        
        total_basis_wt = np.sum([panel.basis_wt for panel in self.panel_objs])

        tl_df = pd.DataFrame()
        
        for i, freq in enumerate(freq_bands):
            w = 2 * np.pi * freq
            k = w / _c_air

            if freq < f_alpha:   # f < f0
                tl = 10 * np.log10(1 + (w * total_basis_wt/(3.6 * _c_air * _rho_air))**2)
            elif freq < f_l:   # f0 < f < f_l
                tl = np.sum([panel.stl_at_frequency(freq) for panel in self.panel_objs]) 
                tl += 20 * np.log10(2 * k * self.absorber_objs[0].thickness)
            else:  # f > f_l
                tl = np.sum([panel.stl_at_frequency(freq) for panel in self.panel_objs])
                tl += self.absorber_objs[0].stl_at_frequency(freq)
                

            df = pd.DataFrame(data={'Frequency': [freq], 'Transmission_Loss': [tl]}, index=[i])
            tl_df = pd.concat([tl_df, df])
        
        return tl_df
    
    
    def triple_panel_loss(self, freq_bands):
        
        self._calculate_air_mass_resonance_frequency()
        self._calculate_standing_wave_frequency()
        
        f_alpha = self.air_mass_resonance_frequency[0]
        f_beta = self.air_mass_resonance_frequency[1]
        
        f_l = min(self.standing_wave_frequency[0], self.standing_wave_frequency[1])
        
        
        R2 = (f_alpha / f_beta)**2
        d1 = self.absorber_objs[0].thickness
        d2 = self.absorber_objs[1].thickness
        d_total = d1 + d2
        K1 = self.absorber_objs[0].bulk_modulus()
        K2 = self.absorber_objs[1].bulk_modulus()
        D = 2 * np.pi * np.sqrt((K1 * d1 + K2 * d2) / d_total)
        D_air = 2 * np.pi * np.sqrt(_rho_air * _c_air**2)
        M1 = self.panel_objs[0].basis_wt
        M2 = self.panel_objs[1].basis_wt
        M3 = self.panel_objs[2].basis_wt
        M = M1 + M2 + M3
        f0 = D * np.sqrt(4 / (M * d_total))
        f1_bar = D * np.sqrt((M1 + M3) / (M1 * M3 * d_total))
        A_f1 = 1 + 2 * R2 * ((f1_bar**2 - f_beta**2) / f0**2)
        t_coef = _rho_air * _c_air * f_alpha / (M * (1-R2) * abs(A_f1))
#         print(f"R2 = {R2}\nd_total = {d_total}\nK1 = {K1} \nK2 = {K2} \nD = {D} \nD_air = {D_air} \n \
#         M = {M} \nf0 = {f0} \nf1_bar = {f1_bar} \nA_f1 = {A_f1}")
        
        tl_df = pd.DataFrame()
        
        for i, freq in enumerate(freq_bands):
            w = 2 * np.pi * freq
            k = w / _c_air

            if freq < f_alpha:   # f < f0
                tl = 10 * np.log10(1 + (w * M/(3.6 * _c_air * _rho_air))**2)
            elif freq < f_beta:
                t_coef /= freq**2
                tl = 10 * np.log10(1 / t_coef)
            elif freq < f_l:   # f0 < f < f_l (lowest of cavity resonance frequency)
                tl = np.sum([panel.stl_at_frequency(freq) for panel in self.panel_objs])
                tl += 20 * np.log10(2 * k * self.absorber_objs[0].thickness)
            else:  # f > f_l
                tl = np.sum([panel.stl_at_frequency(freq) for panel in self.panel_objs])
                tl += np.sum([absorber.stl_at_frequency(freq) for absorber in self.absorber_objs])
                

            df = pd.DataFrame(data={'Frequency': [freq], 'Transmission_Loss': [tl]}, index=[i])
            tl_df = pd.concat([tl_df, df])
        
        return tl_df
    
    
    def combined_transmission_loss(self, frequency_range):
        """
        Calculates the transmission loss of the structure for a given frequency range.

    Parameters
    ----------
    frequency_range : tuple or list, optional
        A tuple or list specifying the minimum and maximum frequencies, in Hz. If not provided, the default frequency
        range is from 20 Hz to 8000 Hz.

        Returns
        -------
        tl_df : pandas.DataFrame
        DataFrame containing the calculated transmission loss for each frequency band.
        """
        
        if (len(frequency_range) > 2) or (len(frequency_range) < 1):
            min_freq = 20
            max_freq = 8000
        elif len(frequency_range) == 2:
            min_freq = frequency_range[0]
            max_freq = frequency_range[1]
        
        freq_bands = list(ac.bands.third(min_freq, max_freq))
        
        if self.n_panels == 1:
            tl_df = self.panel_objs[0].transmission_loss_range(frequency_range)
                
        elif self.n_panels == 2:
            tl_df = self.double_panel_loss(freq_bands)
            
        else:  #self.n_panels == 3
            tl_df = self.triple_panel_loss(freq_bands)
            
        tl_df['Transmission_Loss'] = tl_df['Transmission_Loss'].apply(lambda x: np.round(x, 0))
        
        return tl_df
        





class OITC:
    """
    Class for calculating OITC (Outdoor-Indoor Transmission Class) rating.
    """

    def __init__(self, tl_df):
        """
        Initialize the OITC object.

        Args:
            tl_df (pandas DataFrame): DataFrame containing transmission loss values.
                                     The DataFrame should have a 'Frequency' column
                                     and a 'Transmission_Loss' column.
        """
        self.tl_df = tl_df
        min_freq = min(tl_df['Frequency'])
        max_freq = max(tl_df['Frequency'])
        self.freq_bands = list(ac.bands.third(min_freq, max_freq))

    def estimate_reference_source_spectrum(self):
        """
        Estimate the reference source spectrum from the 1/3 octave center frequencies.

        Returns:
            A pandas Series of reference source spectrum in dB.
        """
        ref_dB = -10.62856656 * np.log10(self.freq_bands) + 121.8467968
        ref_dB = np.round(ref_dB, 0)
        return ref_dB

    def estimate_a_weighting(self):
        """
        Estimate the A-weighting of the transmission loss values.

        Returns:
            A pandas Series of A-weighted transmission loss values in dB.
        """
        log_f13 = np.log10(self.freq_bands)
        a = -196.081218056628
        b = 178.880163703195
        c = -66.1671829289788
        d = 12.6057562121707
        e = -1.05580233283827
        
        a_weighting = a + b * log_f13 + c * (log_f13**2) + d * (log_f13**3) + e * (log_f13**4)
        a_weighting = np.round(a_weighting, 1)
        return a_weighting

    def rating(self):
        """
        Calculate the OITC rating.

        Returns:
            The OITC rating in dB.
        """
        W_f = (self.estimate_reference_source_spectrum() + self.estimate_a_weighting()) / 10
        D_f = self.tl_df['Transmission_Loss']/10
        WD_f = (W_f - D_f)
        
        ref_rating = 10 * np.log10(np.sum(10**(W_f)))
        oitc_rating = ref_rating - 10 * np.log10(np.sum(10**WD_f))
        oitc_rating = np.round(oitc_rating, 1)
        
#         print(f"ref rating = {ref_rating}")
        
        return oitc_rating


# ### Public Functions

def stratified_sample_df(df, col, n_samples):
    """
    Stratifies the dataframe based on the given column(s) and samples a fixed number of rows from each stratum.

    Args:
        df (DataFrame): The input dataframe.
        col (str or list): The column(s) to stratify the dataframe.
        n_samples (int): The number of samples to be selected from each stratum.

    Returns:
        DataFrame: A stratified dataframe containing the sampled rows.

    """
    n = min(n_samples, df[col].value_counts().min())
    stratified_df = df.groupby(col).apply(lambda x: x.sample(n))
    stratified_df = stratified_df.reset_index(drop=True)
    return stratified_df


def create_material_sampling_lists(panels_df, absorbers_df):
    """
    Creates material sampling lists by generating quantile bins for numeric stratification columns.

    Args:
        panels_df (DataFrame): The dataframe containing panel materials.
        absorbers_df (DataFrame): The dataframe containing absorber materials.

    Returns:
        tuple: A tuple containing the list of panel dataframes and the list of absorber dataframes.

    """
    # generate quantile bins for numeric stratification columns
    panels_df['t_bins'] = pd.qcut(panels_df['Thickness'], 5)
    panels_df['m_bins'] = pd.qcut(panels_df['ElasticModulus'], 5)
    panels_df['d_bins'] = pd.qcut(panels_df['Density'], 5)

    absorbers_df['t_bins'] = pd.qcut(absorbers_df['Thickness'], 5)
    absorbers_df['m_bins'] = pd.qcut(absorbers_df['AirFlowResistivity'], 5)
    absorbers_df['d_bins'] = pd.qcut(absorbers_df['Density'], 5)

    # List of panel and absorber dataframes
    panel_df_list = []
    absorber_df_list = []
    for i in range(_max_panels):
        abs_sample = stratified_sample_df(absorbers_df, ['Category', 't_bins', 'm_bins', 'd_bins'], 15)

        if i > 0:
            panel_sample = stratified_sample_df(panels_df[panels_df['Category'] != 'Masonry'],
                                                 ['Category', 'm_bins', 'd_bins', 't_bins'], 15)
        else: #masonry only in first panel
            panel_sample = stratified_sample_df(panels_df,['Category', 'm_bins', 'd_bins', 't_bins'], 15)

        #fix the column name with funny characters
        column_names = abs_sample.columns[abs_sample.columns.str.contains("Name")]
        abs_sample = abs_sample.rename(columns={column_names[0]: "Name"})

        abs_sample.drop(columns=abs_sample.columns[abs_sample.columns.str.contains("bins")], inplace=True)
        panel_sample.drop(columns=panel_sample.columns[panel_sample.columns.str.contains("bins")], inplace=True)

        panel_df_list.append(panel_sample)
        absorber_df_list.append(abs_sample)

    return panel_df_list, absorber_df_list


def preset_result_columns():
    """
    Defines the preset columns for the result dataframe.

    Returns:
        tuple: A tuple containing the feature columns and the transmission loss columns.

    """
    feature_columns = []
    panel_columns = ['Density', 'Thickness_mm', 'ElasticModulusGPa', 'PoissonRatio', 'DampingRatio']
    absorber_columns = ['Density', 'MaterialDensity', 'Thickness_mm', 'SolidElasticModulusGPa',
                        'SolidPoissonRatio', 'DampingRatio', 'AirFlowResistivity', 'Porosity', 'Tortuosity']
    for i in range(_max_panels):
        for col_name in panel_columns:
            feature_columns.append('Panel' + str(i+1) + "_" + col_name)

    for i in range(_max_panels):
        for col_name in absorber_columns:
            feature_columns.append('Absorber' + str(i+1) + "_" + col_name)

    tl_columns = ['Frequency', 'Transmission_Loss']

    return feature_columns, tl_columns


def generate_frame_structure(panel_df_list, absorber_df_list, n_panels, n_absorbers):
    """
    Generates the frame structure dictionary with randomly selected panels and absorbers.

    Args:
        panel_df_list (list): The list of panel dataframes.
        absorber_df_list (list): The list of absorber dataframes.
        n_panels (int): The number of panels in the frame structure.
        n_absorbers (int): The number of absorbers in the frame structure.

    Returns:
        list: A list containing the property dataframe and the frame structure dictionary.

    """
    # Create dictionary of structure with randomly selected panels and absorbers
    frame_structure_dict = {}
    panels_dict = {}
    property_df = pd.DataFrame()
    for i in range(n_panels):
        panel_df_i = panel_df_list[i].sample().iloc[0]
        panels_dict[f"Panel{i+1}"] = panel_df_i
        #add data to property dataframe
        panel_df_i = pd.DataFrame(panel_df_i).transpose()
        panel_df_i['basis_wt'] = panel_df_i['Density'].multiply(panel_df_i['Thickness'])
        panel_df_i = panel_df_i.add_prefix(f"Panel{i+1}_")
        property_df = pd.concat([property_df, panel_df_i.reset_index(drop=True)], axis=1)

    absorbers_dict = {}
    for i in range(n_absorbers):
        absorber_df_i = absorber_df_list[i].sample().iloc[0]
        absorbers_dict[f"Absorber{i+1}"] = absorber_df_i   #f"Absorber_{i+1}"
        absorber_df_i = pd.DataFrame(absorber_df_i).transpose()
        absorber_df_i['basis_wt'] = absorber_df_i['Density'].multiply(absorber_df_i['Thickness'])
        absorber_df_i = absorber_df_i.add_prefix(f"Absorber{i+1}_")
        property_df = pd.concat([property_df, absorber_df_i.reset_index(drop=True)], axis=1)

    frame_structure_dict = {"panels": panels_dict, "absorbers": absorbers_dict}

    return [property_df, frame_structure_dict]


def concatenate_property_tl_dataframes(mp, tl_df, property_df, n_absorbers):
    """
    Concatenates the property set to the transmission loss dataframe.

    Args:
        mp (MultiPanel): The MultiPanel object representing the frame structure.
        tl_df (DataFrame): The transmission loss dataframe.
        property_df (DataFrame): The property dataframe.
        n_absorbers (int): The number of absorbers in the frame structure.

    Returns:
        DataFrame: The concatenated dataframe containing property and transmission loss data.

    """
    # add porosity and tortuosity to property data set
    for i in range(n_absorbers):
        property_df[f"Absorber{i+1}_Porosity"] = np.round(mp.absorber_objs[i].porosity, 4)
        property_df[f"Absorber{i+1}_Tortuosity"] = np.round(mp.absorber_objs[i].tortuosity(), 3)

    #convert elastic modulus from Pa to GPa - dividing by 1e9
    modulus_columns = property_df.columns[property_df.columns.str.contains("Modulus")]
    property_df[modulus_columns] = property_df[modulus_columns].apply(lambda x: np.round(x.divide(1e9), 4), axis=1)
    property_df[[c + 'GPa' for c in modulus_columns]] = property_df[modulus_columns].add_suffix('GPa')

    #convert thickness from m to mm - multiplying by 1e3
    thickness_columns = property_df.columns[property_df.columns.str.contains("Thickness")]
    property_df[thickness_columns] = property_df[thickness_columns].apply(
        lambda x: np.round(x.multiply(1e3), 1), axis=1)
    property_df[[c + '_mm' for c in thickness_columns]] = property_df[thickness_columns].add_suffix('_mm')

    # Repeat the values of the first row of the property DataFrame for all rows in the concatenated DataFrame
    property_row = property_df.iloc[0].values
    property_row_repeated = property_row.repeat(tl_df.shape[0])
    property_row_reshaped = property_row_repeated.reshape(property_df.shape[1], -1).T
    property_repeated_df = pd.DataFrame(property_row_reshaped, columns=property_df.columns)

    # Concatenate the repeated property DataFrame with the transmission loss DataFrame
    concatenated_df = pd.concat([property_repeated_df.reset_index(drop=True),
                                 tl_df.reset_index(drop=True)], axis=1)

    return concatenated_df


def main():
    """
    The main function that executes the simulation.

    """
    panels_df = pd.read_csv('Materials/panel_materials.csv', encoding="ISO-8859-1")
    absorbers_df = pd.read_csv('Materials/absorber_materials.csv', encoding="ISO-8859-1")

    # Loop 1000 to 10,000 times
    n_structures = 1000

    feature_cols, tl_cols = preset_result_columns()

    STL_cols = feature_cols + tl_cols
    OITC_cols = feature_cols + ['Total_Basis_Wt', 'OITC_Rating', 'Sp_OITC_Rating']

    OITC_df = pd.DataFrame(columns=OITC_cols)
    STL_df = pd.DataFrame(columns=STL_cols)

    min_freq = 20
    max_freq = 8000
    frequency_range = [min_freq, max_freq]
    panel_df_list, absorber_df_list = create_material_sampling_lists(panels_df, absorbers_df)

    last_write = 0
    for i in range(n_structures):
        # Randomly choose number of panels and absorbers in each structure
        n_panels = random.randint(1, len(panel_df_list))
        n_absorbers = random.randint(n_panels-1, n_panels)
        # print(f"panels = {n_panels}\nabsorbers = {n_absorbers}")

        # Create dictionary of structure with randomly selected panels and absorbers
        property_df, structure_dict = generate_frame_structure(panel_df_list, absorber_df_list, n_panels, n_absorbers)

        # create structure and calculate transmission loss
        mp = MultiPanel(structure_dict)
        tl_df = mp.combined_transmission_loss(frequency_range)

        property_df = property_df.fillna(value=0)

        concat_tl_df = concatenate_property_tl_dataframes(mp, tl_df, property_df, n_absorbers)
        STL_df = pd.concat([STL_df, concat_tl_df]).reset_index(drop=True).fillna(value=0)
        STL_df = STL_df[STL_cols]

        # Calculate OITC rating for this structure
        property_df['OITC_Rating'] = OITC(tl_df).rating()

        # Calculate specific OITC rating for this structure
        basis_wt_columns = property_df.columns[property_df.columns.str.contains("basis_wt")]
        property_df['Total_Basis_Wt'] = property_df[basis_wt_columns].apply(lambda x: np.round(x.sum(), 2), axis=1)
        property_df['Sp_OITC_Rating'] = property_df[['OITC_Rating', 'Total_Basis_Wt']].apply(
            lambda x: np.round(10 * np.log10(10**(0.1 * x['OITC_Rating']) / x['Total_Basis_Wt']), 1), axis=1)

        OITC_df = pd.concat([OITC_df, property_df]).reset_index(drop=True).fillna(value=0)
        OITC_df = OITC_df[OITC_cols]

        if (i+1) % 1000 == 0:
            STL_df.to_csv(f"Materials/STL_simulation_data_{last_write+1}_to_{i+1}.csv", index=False,
                          chunksize=500)
            OITC_df.to_csv(f"Materials/OITC_simulation_data_{last_write+1}_to_{i+1}.csv", index=False,
                           chunksize=500)
            last_write = i + 1
            OITC_df = pd.DataFrame(columns=OITC_cols)
            STL_df = pd.DataFrame(columns=STL_cols)
            print(f"last_write = {i + 1}")

    # --------- end ------------


if __name__ == "__main__":
    main()