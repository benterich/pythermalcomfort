import csv
import datetime as dt

# Set up logging with a level of WARNING
import os
import re

import numpy as np

from pythermalcomfort.__init__ import __version__
from pythermalcomfort.jos3_functions import construction as cons
from pythermalcomfort.jos3_functions import matrix
from pythermalcomfort.jos3_functions import thermoregulation as threg
from pythermalcomfort.jos3_functions.construction import (
    _to17array,
    validate_body_parameters,
    calculate_operative_temp_when_pmv_is_zero,
)
from pythermalcomfort.jos3_functions.matrix import (
    BODY_NAMES,
    INDEX,
    NUM_NODES,
    VINDEX,
    remove_body_name,
)
from pythermalcomfort.jos3_functions.parameters import ALL_OUT_PARAMS, Default


class JOS3:
    """JOS-3 model simulates human thermal physiology including skin temperature, core
    temperature, sweating rate, etc. for the whole body and 17 local body parts.

    This model was developed at Shin-ichi Tanabe Laboratory, Waseda University
    and was derived from 65 Multi-Node model (https://doi.org/10.1016/S0378-7788(02)00014-2)
    and JOS-2 model (https://doi.org/10.1016/j.buildenv.2013.04.013).

    To use this model, create an instance of the JOS3 class with optional body parameters
    such as body height, weight, age, sex, etc.

    Environmental conditions such as air temperature, mean radiant temperature, air velocity, etc.
    can be set using the setter methods. (ex. X.tdb, X.tr, X.v)
    If you want to set the different conditions in each body part, set them
    as a 17 lengths of list, dictionary, or numpy array format.

    List or numpy array format input must be 17 lengths and means the order of "head", "neck", "chest",
    "back", "pelvis", "left_shoulder", "left_arm", "left_hand", "right_shoulder", "right_arm",
    "right_hand", "left_thigh", "left_leg", "left_foot", "right_thigh", "right_leg" and "right_foot".

    The model output includes local and mean skin temperature, local core temperature,
    local and mean skin wettedness, and heat loss from the skin etc.
    The model output can be accessed using the `dict_results()` method and be converted to a csv file
    using the `to_csv` method.
    Each output parameter also can be accessed using getter methods.
    (ex. X.t_skin, X.t_skin_mean, X.t_core)

    If you use this package, please cite us as follows and mention the version of pythermalcomfort used:
    Y. Takahashi, A. Nomoto, S. Yoda, R. Hisayama, M. Ogata, Y. Ozeki, S. Tanabe,
    Thermoregulation Model JOS-3 with New Open Source Code, Energy & Buildings (2020),
    doi: https://doi.org/10.1016/j.enbuild.2020.110575

    Note: To maintain consistency in variable names for pythermalcomfort,
    some variable names differ from those used in the original paper.

    Attributes
    ----------
    tdb : float or list of floats
        Dry bulb air temperature.
    tr : float or list of floats
        Mean radiant temperature.
    to : float or list of floats
        Operative temperature.
    rh : float or list of floats
        Relative humidity.
    v : float or list of floats
        Air velocity.
    clo : float or list of floats
        Clothing insulation.
    posture : str
        Posture of the subject.
    par : float
        Physical activity ratio.
    body_temp : numpy.ndarray
        Body temperature.
    bsa : numpy.ndarray
        Body surface area.
    r_t : numpy.ndarray
        Radiative heat transfer coefficient.
    r_et : numpy.ndarray
        Evaporative heat transfer coefficient.
    w : numpy.ndarray
        Skin wettedness.
    w_mean : float
        Mean skin wettedness.
    t_skin_mean : float
        Mean skin temperature.
    t_skin : numpy.ndarray
        Skin temperature.
    t_core : numpy.ndarray
        Core temperature.
    t_cb : numpy.ndarray
        Central blood temperature.
    t_artery : numpy.ndarray
        Arterial blood temperature.
    t_vein : numpy.ndarray
        Venous blood temperature.
    t_superficial_vein : numpy.ndarray
        Superficial venous blood temperature.
    t_muscle : numpy.ndarray
        Muscle temperature.
    t_fat : numpy.ndarray
        Fat temperature.
    body_names : list of str
        Names of the body parts.
    results : dict
        Dictionary of the simulation results.
    bmr : float
        Basal metabolic rate.
    version : str
        Version of the JOS3 model.

    Returns
    -------
    cardiac_output :
        cardiac output (the sum of the whole blood flow) [L/h]
    cycle_time :
        the counts of executing one cycle calculation [-]
    dt :
        time step [sec]
    pythermalcomfort_version :
        version of pythermalcomfort [-]
    q_res :
        heat loss by respiration [W]
    q_skin2env :
        total heat loss from the skin (each body part) [W]
    q_thermogenesis_total :
        total thermogenesis of the whole body [W]
    simulation_time :
        simulation times [sec]
    t_core :
        core temperature (each body part) [°C]
    t_skin :
        skin temperature (each body part) [°C]
    t_skin_mean :
        mean skin temperature [°C]
    w :
        skin wettedness (each body part) [-]
    w_mean :
        mean skin wettedness [-]
    weight_loss_by_evap_and_res :
        weight loss by the evaporation and respiration of the whole body [g/sec]
    OPTIONAL PARAMETERS :
        the paramters listed below are returned if ex_output = "all"
    age :
        age [years]
    bf_ava_foot :
        AVA blood flow rate of one foot [L/h]
    bf_ava_hand :
        AVA blood flow rate of one hand [L/h]
    bf_core :
        core blood flow rate (each body part) [L/h]
    bf_fat :
        fat blood flow rate (each body part) [L/h]
    bf_muscle :
        muscle blood flow rate (each body part) [L/h]
    bf_skin :
        skin blood flow rate (each body part) [L/h]
    bsa :
        body surface area (each body part) [m2]
    clo :
        clothing insulation (each body part) [clo]
    e_max :
        maximum evaporative heat loss from the skin (each body part) [W]
    e_skin :
        evaporative heat loss from the skin (each body part) [W]
    e_sweat :
        evaporative heat loss from the skin by only sweating (each body part) [W]
    fat :
        body fat rate [%]
    height :
        body height [m]
    name :
        name of the model [-]
    par :
        physical activity ratio [-]
    q_bmr_core :
        core thermogenesis by basal metabolism (each body part) [W]
    q_bmr_fat :
        fat thermogenesis by basal metabolism (each body part) [W]
    q_bmr_muscle :
        muscle thermogenesis by basal metabolism (each body part) [W]
    q_bmr_skin :
        skin thermogenesis by basal metabolism (each body part) [W]
    q_nst :
        core thermogenesis by non-shivering (each body part) [W]
    q_res_latent :
        latent heat loss by respiration (each body part) [W]
    q_res_sensible :
        sensible heat loss by respiration (each body part) [W]
    q_shiv :
        core or muscle thermogenesis by shivering (each body part) [W]
    q_skin2env_latent :
        latent heat loss from the skin (each body part) [W]
    q_skin2env_sensible :
        sensible heat loss from the skin (each body part) [W]
    q_thermogenesis_core :
        core total thermogenesis (each body part) [W]
    q_thermogenesis_fat :
        fat total thermogenesis (each body part) [W]
    q_thermogenesis_muscle :
        muscle total thermogenesis (each body part) [W]
    q_thermogenesis_skin :
        skin total thermogenesis (each body part) [W]
    q_work :
        core or muscle thermogenesis by work (each body part) [W]
    r_et :
        total clothing evaporative heat resistance (each body part) [(m2*kPa)/W]
    r_t :
        total clothing heat resistance (each body part) [(m2*K)/W]
    rh :
        relative humidity (each body part) [%]
    sex :
        sex [-]
    t_artery :
        arterial temperature (each body part) [°C]
    t_cb :
        central blood temperature [°C]
    t_core_set :
        core set point temperature (each body part) [°C]
    t_fat :
        fat temperature (each body part) [°C]
    t_muscle :
        muscle temperature (each body part) [°C]
    t_skin_set :
        skin set point temperature (each body part) [°C]
    t_superficial_vein :
        superficial vein temperature (each body part) [°C]
    t_vein :
        vein temperature (each body part) [°C]
    tdb :
        dry bulb air temperature (each body part) [°C]
    to :
        operative temperature (each body part) [°C]
    tr :
        mean radiant temperature (each body part) [°C]
    v :
        air velocity (each body part) [m/s]
    weight :
        body weight [kg]

    Examples
    --------
    Build a model and set a body built
    Create an instance of the JOS3 class with optional body parameters such as body height, weight, age, sex, etc.

    .. code-block:: python

        >>> import numpy as np
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> import os
        >>> from pythermalcomfort.models import JOS3
        >>> from pythermalcomfort.jos3_functions.utilities import local_clo_typical_ensembles
        >>>
        >>> model = JOS3(
        >>>     height=1.7,
        >>>     weight=60,
        >>>     fat=20,
        >>>     age=30,
        >>>     sex="male",
        >>>     bmr_equation="japanese",
        >>>     bsa_equation="fujimoto",
        >>>     ex_output="all",
        >>> )
        >>> # Set environmental conditions such as air temperature, mean radiant temperature using the setter methods.
        >>> # Set the first condition
        >>> # Environmental parameters can be input as int, float, list, dict, numpy array format.
        >>> model.tdb = 28  # Air temperature [°C]
        >>> model.tr = 30  # Mean radiant temperature [°C]
        >>> model.rh = 40  # Relative humidity [%]
        >>> model.v = np.array( # Air velocity [m/s]
        >>>     [
        >>>         0.2,  # head
        >>>         0.4,  # neck
        >>>         0.4,  # chest
        >>>         0.1,  # back
        >>>         0.1,  # pelvis
        >>>         0.4,  # left shoulder
        >>>         0.4,  # left arm
        >>>         0.4,  # left hand
        >>>         0.4,  # right shoulder
        >>>         0.4,  # right arm
        >>>         0.4,  # right hand
        >>>         0.1,  # left thigh
        >>>         0.1,  # left leg
        >>>         0.1,  # left foot
        >>>         0.1,  # right thigh
        >>>         0.1,  # right leg
        >>>         0.1,  # right foot
        >>>     ]
        >>> )
        >>> model.clo = local_clo_typical_ensembles["briefs, socks, undershirt, work jacket, work pants, safety shoes"]["local_body_part"]
        >>> # par should be input as int, float.
        >>> model.par = 1.2  # Physical activity ratio [-], assuming a sitting position
        >>> # posture should be input as int (0, 1, or 2) or str ("standing", "sitting" or "lying").
        >>> # (0="standing", 1="sitting" or 2="lying")
        >>> model.posture = "sitting"  # Posture [-], assuming a sitting position
        >>>
        >>> # Run JOS-3 model
        >>> model.simulate(
        >>>     times=30,  # Number of loops of a simulation
        >>>     dtime=60,  # Time delta [sec]. The default is 60.
        >>> )  # Exposure time = 30 [loops] * 60 [sec] = 30 [min]
        >>> # Set the next condition (You only need to change the parameters that you want to change)
        >>> model.to = 20  # Change operative temperature
        >>> model.v = { # Air velocity [m/s], assuming to use a desk fan
        >>>     'head' : 0.2,
        >>>     'neck' : 0.4,
        >>>     'chest' : 0.4,
        >>>     'back': 0.1,
        >>>     'pelvis' : 0.1,
        >>>     'left_shoulder' : 0.4,
        >>>     'left_arm' : 0.4,
        >>>     'left_hand' : 0.4,
        >>>     'right_shoulder' : 0.4,
        >>>     'right_arm' : 0.4,
        >>>     'right_hand' : 0.4,
        >>>     'left_thigh' : 0.1,
        >>>     'left_leg' : 0.1,
        >>>     'left_foot' : 0.1,
        >>>     'right_thigh' : 0.1,
        >>>     'right_leg' : 0.1,
        >>>     'right_foot' : 0.1
        >>>     }
        >>> # Run JOS-3 model
        >>> model.simulate(
        >>>     times=60,  # Number of loops of a simulation
        >>>     dtime=60,  # Time delta [sec]. The default is 60.
        >>> )  # Additional exposure time = 60 [loops] * 60 [sec] = 60 [min]
        >>> # Set the next condition (You only need to change the parameters that you want to change)
        >>> model.tdb = 30  # Change air temperature [°C]
        >>> model.tr = 35  # Change mean radiant temperature [°C]
        >>> # Run JOS-3 model
        >>> model.simulate(
        >>>     times=30,  # Number of loops of a simulation
        >>>     dtime=60,  # Time delta [sec]. The default is 60.
        >>> )  # Additional exposure time = 30 [loops] * 60 [sec] = 30 [min]
        >>> # Show the results
        >>> df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
        >>> df[["t_skin_mean", "t_skin_head", "t_skin_chest", "t_skin_left_hand"]].plot()  # Plot time series of local skin temperature.
        >>> plt.legend(["Mean", "Head", "Chest", "Left hand"])  # Reset the legends
        >>> plt.ylabel("Skin temperature [°C]")  # Set y-label as 'Skin temperature [°C]'
        >>> plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
        >>> plt.show()  # Show the plot
    """

    def __init__(
        self,
        height: float = Default.height,
        weight: float = Default.weight,
        fat: float = Default.body_fat,
        age: int = Default.age,
        sex: float = Default.sex,
        ci: float = Default.cardiac_index,
        bmr_equation: str = Default.bmr_equation,
        bsa_equation: str = Default.bsa_equation,
        ex_output=None,
    ):
        """Initialize a new instance of the JOS3 class, which models and simulates various
        physiological parameters related to human thermoregulation.

        This class uses mathematical models to calculate and predict body temperature,
        basal metabolic rate, body surface area, and other related parameters.

        Parameters
        ----------
        height : float, optional
            Body height in meters. Default is 1.72.
        weight : float, optional
            Body weight in kilograms. Default is 74.43.
        fat : float, optional
            Fat percentage. Default is 15.
        age : int, optional
            Age in years. Default is 20.
        sex : str, optional
            Sex ("male" or "female"). Default is "male".
        ci : float, optional
            Cardiac index in liters per minute per square meter. Default is 2.6432.
        bmr_equation : str, optional
            The equation used to calculate basal metabolic rate (BMR). Options are "harris-benedict"
            for Caucasian data (DOI: doi.org/10.1073/pnas.4.12.370) or "japanese" for Ganpule's equation
            (DOI: doi.org/10.1038/sj.ejcn.1602645). Default is "harris-benedict".
        bsa_equation : str, optional
            The equation used to calculate body surface area (BSA). Choose one from pythermalcomfort.utilities.BodySurfaceAreaEquations. Default is "dubois".
        ex_output : None or "all", optional
            Additional output parameters. If None, no extra output is provided. If "all", all possible
            outputs are included. Default is None.

        Returns
        -------
        None.

        Examples
        --------
        Create an instance of the JOS3 class with optional body parameters:

        .. code-block:: python

            jos3_model = JOS3(height=1.75, weight=70, age=25, sex="female")
        """
        # Version of pythermalcomfort
        version_string = (
            __version__  # get the current version of pythermalcomfort package
        )
        version_number_string = re.findall(r"\d+\.\d+\.\d+", version_string)[0]
        self._version = version_number_string  # (ex. 'X.Y.Z')

        # validate body parameters
        validate_body_parameters(height=height, weight=weight, age=age, body_fat=fat)

        # Initialize basic attributes
        self._height = height
        self._weight = weight
        self._fat = fat
        self._sex = sex
        self._age = age
        self._ci = ci
        self._bmr_equation = bmr_equation
        self._bsa_equation = bsa_equation
        self._ex_output = ex_output

        # Calculate body surface area (bsa) rate
        self._bsa_rate = cons.bsa_rate(height, weight, bsa_equation)

        # Calculate local body surface area
        self._bsa = cons.local_bsa(height, weight, bsa_equation)

        # Calculate basal blood flow (BFB) rate [-]
        self._bfb_rate = cons.bfb_rate(height, weight, bsa_equation, age, ci)

        # Calculate thermal conductance (CDT) [W/K]
        self._cdt = cons.conductance(height, weight, bsa_equation, fat)

        # Calculate thermal capacity [J/K]
        self._cap = cons.capacity(height, weight, bsa_equation, age, ci)

        # Set initial core and skin temperature set points [°C]
        self.cr_set_point = np.ones(17) * Default.core_temperature
        self.sk_set_point = np.ones(17) * Default.skin_temperature

        # Initialize body temperature [°C]
        self._t_body = np.ones(NUM_NODES) * Default.other_body_temperature

        # Initialize environmental conditions and other factors
        # (Default values of input conditions)
        self._ta = np.ones(17) * Default.dry_bulb_air_temperature
        self._tr = np.ones(17) * Default.mean_radiant_temperature
        self._rh = np.ones(17) * Default.relative_humidity
        self._va = np.ones(17) * Default.air_speed
        self._clo = np.ones(17) * Default.clothing_insulation
        self._iclo = np.ones(17) * Default.clothing_vapor_permeation_efficiency
        self._par = Default.physical_activity_ratio
        self._posture = Default.posture
        self._hc = None  # Convective heat transfer coefficient
        self._hr = None  # Radiative heat transfer coefficient
        self.ex_q = np.zeros(NUM_NODES)  # External heat gain
        self._time = dt.timedelta(0)  # Elapsed time
        self._cycle = 0  # Cycle time
        self.model_name = "JOS3"  # Model name
        self.options = {
            "nonshivering_thermogenesis": True,
            "cold_acclimated": False,
            "shivering_threshold": False,
            "limit_dshiv/dt": False,
            "bat_positive": False,
            "ava_zero": False,
            "shivering": False,
        }

        # Set shivering threshold = 0
        threg.PRE_SHIV = 0

        # Initialize history to store model parameters
        self._history = []

        # Set elapsed time and cycle time to 0
        self._time = dt.timedelta(0)  # Elapsed time
        self._cycle = 0  # Cycle time

        # Reset set-point temperature and save the last model parameters
        dictout = self._reset_setpt(par=Default.physical_activity_ratio)
        self._history.append(dictout)

    def _reset_setpt(self, par=Default.physical_activity_ratio):
        """Reset set-point temperatures under steady state calculation. Set- point
        temperatures are hypothetical core or skin temperatures in a thermally neutral
        state when at rest (similar to room set-point temperature for air conditioning).
        This function is used during initialization to calculate the set-point
        temperatures as a reference for thermoregulation. Be careful, input parameters
        (tdb, tr, rh, v, clo, par) and body temperatures are also reset.

        Returns
        -------
        dict
            Parameters of JOS-3 model.
        """
        # Set operative temperature under PMV=0 environment
        # 1 met = 58.15 W/m2
        w_per_m2_to_met = 1 / 58.15  # unit converter W/m2 to met
        met = self.bmr * par * w_per_m2_to_met  # [met]
        self.to = calculate_operative_temp_when_pmv_is_zero(met=met)
        self.rh = Default.relative_humidity
        self.v = Default.air_speed
        self.clo = Default.clothing_insulation
        self.par = par  # Physical activity ratio

        # Steady-calculation
        self.options["ava_zero"] = True
        for _ in range(10):
            dict_out = self._run(dtime=60000, passive=True)

        # Set new set-point temperatures for core and skin
        self.cr_set_point = self.t_core
        self.sk_set_point = self.t_skin
        self.options["ava_zero"] = False

        return dict_out

    def simulate(self, times: int, dtime=60, output: bool = True) -> None:
        """
        Run the JOS-3 model simulation.

        This method executes the JOS-3 model for a specified number of loops, simulating
        human thermoregulation over time. The results of each simulation step can be recorded
        and accessed later.

        Parameters
        ----------
        times : int
            Number of loops of the simulation.
        dtime : int or float, optional
            Time delta in seconds for each simulation step. Default is 60.
        output : bool, optional
            If True, records the parameters at each simulation step. Default is True.

        Returns
        -------
        None

        Examples
        --------
        Create an instance of the JOS3 class and run the simulation:

        .. code-block:: python
            :emphasize-lines: 7

            from pythermalcomfort.models import JOS3

            # Create an instance of the JOS3 class
            jos3_model = JOS3(height=1.75, weight=70, age=25, sex="female")

            # Run the simulation for 30 loops with a time delta of 60 seconds
            jos3_model.simulate(times=30, dtime=60)

            # Access the results
            results = jos3_model.dict_results()
            print(results)
        """
        # Loop through the simulation for the given number of times
        for _ in range(times):
            # Increment the elapsed time by the time delta
            self._time += dt.timedelta(0, dtime)

            # Increment the cycle counter
            self._cycle += 1

            # Execute the simulation step
            dict_data = self._run(dtime=dtime, output=output)

            # If output is True, append the results to the history
            if output:
                self._history.append(dict_data)

    def _run(self, dtime=60, passive=False, output=True):
        """Run a single cycle of the JOS-3 model simulation.

        This method calculates various thermoregulation parameters using the input data,
        such as convective and radiative heat transfer coefficients, operative temperature,
        heat resistance, and blood flow rates. It also computes thermogenesis by shivering
        and non-shivering, basal thermogenesis, and thermogenesis by work. The function
        constructs and solves matrices to determine the new body temperature distribution.

        Parameters
        ----------
        dtime : int or float, optional
            Time step in seconds for the simulation. Default is 60.
        passive : bool, optional
            If True, the set-point temperature for thermoregulation is set to the current
            body temperature, simulating a passive model. Default is False.
        output : bool, optional
            If True, returns a dictionary of the simulation results. Default is True.

        Returns
        -------
        dict
            A dictionary containing the simulation results, including parameters such as
            cycle time, model time, mean skin temperature, skin temperature, core temperature,
            mean skin wettedness, skin wettedness, weight loss by evaporation and respiration,
            cardiac output, total thermogenesis, respiratory heat loss, and total heat loss
            from the skin to the environment. If `ex_output` is set to "all" or a list of keys,
            additional detailed parameters are included.
        """
        # Compute convective and radiative heat transfer coefficient [W/(m2*K)]
        # based on posture, air velocity, air temperature, and skin temperature.
        # Manual setting is possible by setting self._hc and self._hr.
        # Compute heat and evaporative heat resistance [m2.K/W], [m2.kPa/W]

        # Get core and skin temperatures
        tcr = self.t_core
        tsk = self.t_skin

        # Convective and radiative heat transfer coefficients [W/(m2*K)]
        hc = threg.fixed_hc(
            threg.conv_coef(
                self._posture,
                self._va,
                self._ta,
                tsk,
            ),
            self._va,
        )
        hr = threg.fixed_hr(
            threg.rad_coef(
                self._posture,
            )
        )

        # Manually set convective and radiative heat transfer coefficients if necessary
        if self._hc is not None:
            hc = self._hc
        if self._hr is not None:
            hr = self._hr

        # Compute operative temp. [°C], clothing heat and evaporative resistance [m2.K/W], [m2.kPa/W]
        # Operative temp. [°C]
        to = threg.operative_temp(
            self._ta,
            self._tr,
            hc,
            hr,
        )
        # Clothing heat resistance [m2.K/W]
        r_t = threg.dry_r(hc, hr, self._clo)
        # Clothing evaporative resistance [m2.kPa/W]
        r_et = threg.wet_r(hc, self._clo, self._iclo)

        # ------------------------------------------------------------------
        # Thermoregulation
        # 1) Sweating
        # 2) Vasoconstriction, Vasodilation
        # 3) Shivering and non-shivering thermogenesis
        # ------------------------------------------------------------------

        # Compute the difference between the set-point temperature and body temperatures
        # and other thermoregulation parameters.
        # If running a passive model, the set-point temperature of thermoregulation is
        # set to the current body temperature.

        # set-point temperature for thermoregulation
        if passive:
            setpt_cr = tcr.copy()
            setpt_sk = tsk.copy()
        else:
            setpt_cr = self.cr_set_point.copy()
            setpt_sk = self.sk_set_point.copy()

        # Error signal = Difference between set-point and body temperatures
        err_cr = tcr - setpt_cr
        err_sk = tsk - setpt_sk

        # SWEATING THERMOREGULATION
        # Skin wettedness [-], e_skin, e_max, e_sweat [W]
        # Calculate skin wettedness, sweating heat loss, maximum sweating rate, and total sweat rate
        wet, e_sk, e_max, e_sweat = threg.evaporation(
            err_cr,
            err_sk,
            tsk,
            self._ta,
            self._rh,
            r_et,
            self._height,
            self._weight,
            self._bsa_equation,
            self._age,
        )

        # VASOCONSTRICTION, VASODILATION
        # Calculate skin blood flow and basal skin blood flow [L/h]
        bf_skin = threg.skin_blood_flow(
            err_cr,
            err_sk,
            self._height,
            self._weight,
            self._bsa_equation,
            self._age,
            self._ci,
        )

        # Calculate hands and feet's AVA blood flow [L/h]
        bf_ava_hand, bf_ava_foot = threg.ava_blood_flow(
            err_cr,
            err_sk,
            self._height,
            self._weight,
            self._bsa_equation,
            self._age,
            self._ci,
        )
        if self.options["ava_zero"] and passive:
            bf_ava_hand = 0
            bf_ava_foot = 0

        # SHIVERING AND NON-SHIVERING
        # Calculate shivering thermogenesis [W]
        q_shiv = threg.shivering(
            err_cr,
            err_sk,
            tcr,
            tsk,
            self._height,
            self._weight,
            self._bsa_equation,
            self._age,
            self._sex,
            dtime,
            self.options,
        )

        # Calculate non-shivering thermogenesis (NST) [W]
        if self.options["nonshivering_thermogenesis"]:
            q_nst = threg.nonshivering(
                err_sk,
                self._height,
                self._weight,
                self._bsa_equation,
                self._age,
                self.options["cold_acclimated"],
                self.options["bat_positive"],
            )
        else:  # not consider NST
            q_nst = np.zeros(17)

        # ------------------------------------------------------------------
        # Thermogenesis
        # ------------------------------------------------------------------

        # Calculate local basal metabolic rate (BMR) [W]
        q_bmr_local = threg.local_mbase(
            self._height,
            self._weight,
            self._age,
            self._sex,
            self._bmr_equation,
        )
        # Calculate overall basal metabolic rate (BMR) [W]
        q_bmr_total = sum([m.sum() for m in q_bmr_local])

        # Calculate thermogenesis by work [W]
        q_work = threg.local_q_work(q_bmr_total, self._par)

        # Calculate the sum of thermogenesis in core, muscle, fat, skin [W]
        (
            q_thermogenesis_core,
            q_thermogenesis_muscle,
            q_thermogenesis_fat,
            q_thermogenesis_skin,
        ) = threg.sum_m(
            q_bmr_local,
            q_work,
            q_shiv,
            q_nst,
        )
        q_thermogenesis_total = (
            q_thermogenesis_core.sum()
            + q_thermogenesis_muscle.sum()
            + q_thermogenesis_fat.sum()
            + q_thermogenesis_skin.sum()
        )

        # ------------------------------------------------------------------
        # Others
        # ------------------------------------------------------------------
        # Calculate blood flow in core, muscle, fat [L/h]
        bf_core, bf_muscle, bf_fat = threg.cr_ms_fat_blood_flow(
            q_work,
            q_shiv,
            self._height,
            self._weight,
            self._bsa_equation,
            self._age,
            self._ci,
        )

        # Calculate heat loss by respiratory
        p_a = threg.antoine(self._ta) * self._rh / 100
        res_sh, res_lh = threg.resp_heat_loss(
            self._ta[0], p_a[0], q_thermogenesis_total
        )

        # Calculate sensible heat loss [W]
        shl_sk = (tsk - to) / r_t * self._bsa

        # Calculate cardiac output [L/h]
        co = threg.sum_bf(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot)

        # Calculate weight loss rate by evaporation [g/sec]
        wlesk = (e_sweat + 0.06 * e_max) / 2418
        wleres = res_lh / 2418

        # ------------------------------------------------------------------
        # Matrix
        # This code section is focused on constructing and calculating
        # various matrices required for modeling the thermoregulation
        # of the human body.
        # Since JOS-3 has 85 thermal nodes, the determinant of 85*85 is to be solved.
        # ------------------------------------------------------------------

        # Matrix A = Matrix for heat exchange due to blood flow and conduction occurring between tissues
        # (85, 85,) ndarray

        # Calculates the blood flow in arteries and veins for core, muscle, fat, skin,
        # and arteriovenous anastomoses (AVA) in hands and feet,
        # and combines them into two arrays:
        # 1) bf_local for the local blood flow and 2) bf_whole for the whole-body blood flow.
        # These arrays are then combined to form arr_bf.
        bf_art, bf_vein = matrix.vessel_blood_flow(
            bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot
        )
        bf_local = matrix.local_arr(
            bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot
        )
        bf_whole = matrix.whole_body(bf_art, bf_vein, bf_ava_hand, bf_ava_foot)
        arr_bf = np.zeros((NUM_NODES, NUM_NODES))
        arr_bf += bf_local
        arr_bf += bf_whole

        # Adjusts the units of arr_bf from [W/K] to [/sec] and then to [-]
        # by dividing by the heat capacity self._cap and multiplying by the time step dtime.
        arr_bf /= self._cap.reshape((NUM_NODES, 1))  # Change unit [W/K] to [/sec]
        arr_bf *= dtime  # Change unit [/sec] to [-]

        # Performs similar unit conversions for the convective heat transfer coefficient array arr_cdt
        # (also divided by self._cap and multiplied by dtime).
        arr_cdt = self._cdt.copy()
        arr_cdt /= self._cap.reshape((NUM_NODES, 1))  # Change unit [W/K] to [/sec]
        arr_cdt *= dtime  # Change unit [/sec] to [-]

        # Matrix B = Matrix for heat transfer between skin and environment
        arr_b = np.zeros(NUM_NODES)
        arr_b[INDEX["skin"]] += 1 / r_t * self._bsa
        arr_b /= self._cap  # Change unit [W/K] to [/sec]
        arr_b *= dtime  # Change unit [/sec] to [-]

        # Calculates the off-diagonal and diagonal elements of the matrix A,
        # which represents the heat transfer coefficients between different parts of the body,
        # and combines them to form the full matrix A (arrA).
        # Then, the inverse of matrix A is computed (arrA_inv).
        arr_a_tria = -(arr_cdt + arr_bf)

        arr_a_dia = arr_cdt + arr_bf
        arr_a_dia = arr_a_dia.sum(axis=1) + arr_b
        arr_a_dia = np.diag(arr_a_dia)
        arr_a_dia += np.eye(NUM_NODES)

        arr_a = arr_a_tria + arr_a_dia
        arr_a_inv = np.linalg.inv(arr_a)

        # Matrix Q = Matrix for heat generation rate from thermogenesis, respiratory, sweating,
        # and extra heat gain processes in different body parts.

        # Matrix Q [W] / [J/K] * [sec] = [-]
        # Thermogensis
        arr_q = np.zeros(NUM_NODES)
        arr_q[INDEX["core"]] += q_thermogenesis_core
        arr_q[INDEX["muscle"]] += q_thermogenesis_muscle[VINDEX["muscle"]]
        arr_q[INDEX["fat"]] += q_thermogenesis_fat[VINDEX["fat"]]
        arr_q[INDEX["skin"]] += q_thermogenesis_skin

        # Respiratory [W]
        arr_q[INDEX["core"][2]] -= res_sh + res_lh  # chest core

        # Sweating [W]
        arr_q[INDEX["skin"]] -= e_sk

        # Extra heat gain [W]
        arr_q += self.ex_q.copy()

        arr_q /= self._cap  # Change unit [W]/[J/K] to [K/sec]
        arr_q *= dtime  # Change unit [K/sec] to [K]

        # Boundary batrix [°C]
        arr_to = np.zeros(NUM_NODES)
        arr_to[INDEX["skin"]] += to

        # Combines the current body temperature, the boundary matrix, and the heat generation matrix
        # to calculate the new body temperature distribution (arr).
        arr = self._t_body + arr_b * arr_to + arr_q

        # ------------------------------------------------------------------
        # New body temp. [°C]
        # ------------------------------------------------------------------
        self._t_body = np.dot(arr_a_inv, arr)

        # ------------------------------------------------------------------
        # Output parameters
        # ------------------------------------------------------------------
        dict_out = {}
        if output:  # Default output
            dict_out["pythermalcomfort_version"] = self._version
            dict_out["cycle_time"] = self._cycle
            dict_out["simulation_time"] = self._time
            dict_out["dt"] = dtime
            dict_out["t_skin_mean"] = np.round(self.t_skin_mean, 2)
            dict_out["t_skin"] = np.round(self.t_skin, 2)
            dict_out["t_core"] = np.round(self.t_core, 2)
            dict_out["w_mean"] = round(np.average(wet, weights=Default.local_bsa), 2)
            dict_out["w"] = np.round(wet, 2)
            dict_out["weight_loss_by_evap_and_res"] = round(wlesk.sum() + wleres, 5)
            dict_out["cardiac_output"] = round(co, 1)
            dict_out["q_thermogenesis_total"] = round(q_thermogenesis_total, 2)
            dict_out["q_res"] = round(res_sh + res_lh, 2)
            dict_out["q_skin2env"] = np.round(shl_sk + e_sk, 2)

        detail_out = {}
        if self._ex_output and output:
            detail_out["name"] = self.model_name
            detail_out["height"] = self._height
            detail_out["weight"] = self._weight
            detail_out["bsa"] = self._bsa
            detail_out["fat"] = self._fat
            detail_out["sex"] = self._sex
            detail_out["age"] = self._age
            detail_out["t_core_set"] = np.round(setpt_cr, 2)
            detail_out["t_skin_set"] = np.round(setpt_sk, 2)
            detail_out["t_cb"] = round(self.t_cb, 2)
            detail_out["t_artery"] = np.round(self.t_artery)
            detail_out["t_vein"] = np.round(self.t_vein)
            detail_out["t_superficial_vein"] = np.round(self.t_superficial_vein)
            detail_out["t_muscle"] = np.round(self.t_muscle)
            detail_out["t_fat"] = np.round(self.t_fat)
            detail_out["to"] = np.round(to, 2)
            detail_out["r_t"] = np.round(r_t, 3)
            detail_out["r_et"] = np.round(r_et, 3)
            detail_out["tdb"] = np.round(self._ta.copy(), 2)
            detail_out["tr"] = np.round(self._tr.copy(), 2)
            detail_out["rh"] = self._rh.copy()
            detail_out["v"] = self._va.copy()
            detail_out["par"] = self._par
            detail_out["clo"] = self._clo.copy()
            detail_out["e_skin"] = np.round(e_sk, 2)
            detail_out["e_max"] = np.round(e_max, 2)
            detail_out["e_sweat"] = np.round(e_sweat, 2)
            detail_out["bf_core"] = np.round(bf_core, 2)
            detail_out["bf_muscle"] = np.round(bf_muscle[VINDEX["muscle"]], 2)
            detail_out["bf_fat"] = np.round(bf_fat[VINDEX["fat"]], 2)
            detail_out["bf_skin"] = np.round(bf_skin, 2)
            detail_out["bf_ava_hand"] = np.round(bf_ava_hand, 2)
            detail_out["bf_ava_foot"] = np.round(bf_ava_foot, 2)
            detail_out["q_bmr_core"] = np.round(q_bmr_local[0], 2)
            detail_out["q_bmr_muscle"] = np.round(q_bmr_local[1][VINDEX["muscle"]], 2)
            detail_out["q_bmr_fat"] = np.round(q_bmr_local[2][VINDEX["fat"]], 2)
            detail_out["q_bmr_skin"] = np.round(q_bmr_local[3], 2)
            detail_out["q_work"] = np.round(q_work, 2)
            detail_out["q_shiv"] = np.round(q_shiv, 2)
            detail_out["q_nst"] = np.round(q_nst, 2)
            detail_out["q_thermogenesis_core"] = np.round(q_thermogenesis_core, 2)
            detail_out["q_thermogenesis_muscle"] = np.round(
                q_thermogenesis_muscle[VINDEX["muscle"]], 2
            )
            detail_out["q_thermogenesis_fat"] = np.round(
                q_thermogenesis_fat[VINDEX["fat"]], 2
            )
            detail_out["q_thermogenesis_skin"] = np.round(q_thermogenesis_skin, 2)
            dict_out["q_skin2env_sensible"] = np.round(shl_sk, 2)
            dict_out["q_skin2env_latent"] = np.round(e_sk, 2)
            dict_out["q_res_sensible"] = np.round(res_sh, 2)
            dict_out["q_res_latent"] = np.round(res_lh, 2)

        if self._ex_output == "all":
            dict_out.update(detail_out)
        elif isinstance(self._ex_output, list):  # if ex_out type is list
            out_keys = detail_out.keys()
            for key in self._ex_output:
                if key in out_keys:
                    dict_out[key] = detail_out[key]
        return dict_out

    def dict_results(self):
        """
        Get simulation results as a dictionary.

        This method returns the results of the JOS-3 model simulation as a dictionary,
        where each key corresponds to a specific parameter and the value is an array
        containing the time series data for that parameter.

        Returns
        -------
        dict
            A dictionary of the simulation results. Keys are parameter names, and values
            are arrays containing the time series data for each parameter.

        Examples
        --------
        Access the results after running a simulation:

        .. code-block:: python
            :emphasize-lines: 3

            jos3_model = JOS3(height=1.75, weight=70, age=25, sex="female")
            jos3_model.simulate(times=30, dtime=60)
            results = jos3_model.dict_results()
            print(
                results["t_skin_mean"]
            )  # Access the mean skin temperature time series
        """
        if not self._history:
            print("The model has no data.")
            return None

        def check_word_contain(word, *args):
            """Check if word contains *args."""
            bool_filter = False
            for arg in args:
                if arg in word:
                    bool_filter = True
            return bool_filter

        # Set column titles
        # If the values are iter, add the body names as suffix words.
        # If the values are not iter and the single value data, convert it to iter.
        key2keys = {}  # Column keys
        for key, value in self._history[0].items():
            try:
                length = len(value)
                if isinstance(value, str):
                    keys = [key]  # str is iter. Convert to list without suffix
                elif check_word_contain(key, "sve", "sfv", "superficialvein"):
                    keys = [key + "_" + BODY_NAMES[i] for i in VINDEX["sfvein"]]
                elif check_word_contain(key, "ms", "muscle"):
                    keys = [key + "_" + BODY_NAMES[i] for i in VINDEX["muscle"]]
                elif check_word_contain(key, "fat"):
                    keys = [key + "_" + BODY_NAMES[i] for i in VINDEX["fat"]]
                elif length == 17:  # if data contains 17 values
                    keys = [key + "_" + bn for bn in BODY_NAMES]
                else:
                    keys = [key + "_" + BODY_NAMES[i] for i in range(length)]
            except TypeError:  # if the value is not iter.
                keys = [key]  # convert to iter
            key2keys.update({key: keys})

        data = []
        for _i, dictout in enumerate(self._history):
            row = {}
            for key, value in dictout.items():
                keys = key2keys[key]
                if len(keys) == 1:
                    values = [value]  # make list if value is not iter
                else:
                    values = value
                row.update(dict(zip(keys, values)))
            data.append(row)

        out_dict = dict(zip(data[0].keys(), [[] for _ in range(len(data[0].keys()))]))
        for row in data:
            for k in data[0].keys():
                out_dict[k].append(row[k])
        return out_dict

    def to_csv(
        self,
        path: str = None,
        folder: str = None,
        unit: bool = True,
        meaning: bool = True,
    ) -> None:
        """Export results as csv format.

        Parameters
        ----------
        path : str, optional
            Output path. If you don't use the default file name, set a name.
            The default is None.
        folder : str, optional
            Output folder. If you use the default file name with the current time,
            set a only folder path.
            The default is None.
        unit : bool, optional
            Write units in csv file. The default is True.
        meaning : bool, optional
            Write meanings of the parameters in csv file. The default is True.

        Returns
        -------
        None

        Examples
        --------
        >>> from pythermalcomfort.models import JOS3
        >>> model = JOS3()
        >>> model.simulate(60)
        >>> model.to_csv()
        """
        # Use the model name and current time as default output file name
        if path is None:
            now_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = f"{self.model_name}_{now_time}.csv"
            if folder:
                os.makedirs(folder, exist_ok=True)
                path = folder + os.sep + path
        elif not ((path[-4:] == ".csv") or (path[-4:] == ".txt")):
            path += ".csv"

        # Get simulation results as a dictionary
        dict_out = self.dict_results()

        # Get column names, units and meanings
        columns = [k for k in dict_out.keys()]
        units = []
        meanings = []
        for col in columns:
            param, body_name = remove_body_name(col)
            if param in ALL_OUT_PARAMS:
                u = ALL_OUT_PARAMS[param]["unit"]
                units.append(u)

                m = ALL_OUT_PARAMS[param]["meaning"]
                if body_name:
                    # Replace underscores with spaces
                    body_name_with_spaces = body_name.replace("_", " ")
                    meanings.append(m.replace("each body part", body_name_with_spaces))
                else:
                    meanings.append(m)
            else:
                units.append("")
                meanings.append("")

        # Write to csv file
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(list(columns))
            if unit:
                writer.writerow(units)
            if meaning:
                writer.writerow(meanings)
            for i in range(len(dict_out["cycle_time"])):
                row = []
                for k in columns:
                    row.append(dict_out[k][i])
                writer.writerow(row)

    def _set_ex_q(self, tissue, value):
        """Set extra heat gain by tissue name.

        Parameters
        ----------
        tissue : str
            Tissue name. "core", "skin", or "artery".... If you set value to
            head muscle and other segment's core, set "all_muscle".
        value : int, float, array
            Heat gain [W]

        Returns
        -------
        array
            Extra heat gain of model.
        """
        self.ex_q[INDEX[tissue]] = value
        return self.ex_q

    @property
    def tdb(self):
        """Dry-bulb air temperature. The setter accepts int, float, dict, list, ndarray.
        The inputs are used to create a 17-element array. dict should be passed with
        BODY_NAMES as keys.

        Returns
        -------
        ndarray
            A NumPy array of shape (17,).
        """
        return self._ta

    @tdb.setter
    def tdb(self, inp):
        self._ta = _to17array(inp)

    @property
    def tr(self):
        """Tr : numpy.ndarray (17,) Mean radiant temperature [°C]."""
        return self._tr

    @tr.setter
    def tr(self, inp):
        self._tr = _to17array(inp)

    @property
    def to(self):
        """To : numpy.ndarray (17,) Operative temperature [°C]."""
        hc = threg.fixed_hc(
            threg.conv_coef(
                self._posture,
                self._va,
                self._ta,
                self.t_skin,
            ),
            self._va,
        )
        hr = threg.fixed_hr(
            threg.rad_coef(
                self._posture,
            )
        )
        to = threg.operative_temp(
            self._ta,
            self._tr,
            hc,
            hr,
        )
        return to

    @to.setter
    def to(self, inp):
        self._ta = _to17array(inp)
        self._tr = _to17array(inp)

    @property
    def rh(self):
        """Rh : numpy.ndarray (17,) Relative humidity [%]."""
        return self._rh

    @rh.setter
    def rh(self, inp):
        self._rh = _to17array(inp)

    @property
    def v(self):
        """V : numpy.ndarray (17,) Air velocity [m/s]."""
        return self._va

    @v.setter
    def v(self, inp):
        self._va = _to17array(inp)

    @property
    def posture(self):
        """Posture : str Current JOS3 posture."""
        return self._posture

    @posture.setter
    def posture(self, inp):
        if inp == 0:
            self._posture = "standing"
        elif inp == 1:
            self._posture = "sitting"
        elif inp == 2:
            self._posture = "lying"
        elif isinstance(inp, str):
            if inp.lower() == "standing":
                self._posture = "standing"
            elif inp.lower() in ["sitting", "sedentary"]:
                self._posture = "sitting"
            elif inp.lower() in ["lying", "supine"]:
                self._posture = "lying"
        else:
            self._posture = "standing"
            print('posture must be 0="standing", 1="sitting" or 2="lying".')
            print('posture was set "standing".')

    @property
    def clo(self):
        """Clo : numpy.ndarray (17,) Clothing insulation [clo]."""
        return self._clo

    @clo.setter
    def clo(self, inp):
        self._clo = _to17array(inp)

    @property
    def par(self):
        """Par : float Physical activity ratio [-].This equals the ratio of metabolic rate to basal metabolic rate. par of sitting quietly is 1.2."""
        return self._par

    @par.setter
    def par(self, inp):
        self._par = inp

    @property
    def body_temp(self):
        """body_temp : numpy.ndarray (85,) All segment temperatures of JOS-3"""
        return self._t_body

    @body_temp.setter
    def body_temp(self, inp):
        self._t_body = inp.copy()

    @property
    def bsa(self):
        """Bsa : numpy.ndarray (17,) Body surface areas by local body segments [m2]."""
        return self._bsa.copy()

    @property
    def r_t(self):
        """r_t : numpy.ndarray (17,) Dry heat resistances between the skin and ambience areas by local body segments [(m2*K)/W]."""
        hc = threg.fixed_hc(
            threg.conv_coef(
                self._posture,
                self._va,
                self._ta,
                self.t_skin,
            ),
            self._va,
        )
        hr = threg.fixed_hr(
            threg.rad_coef(
                self._posture,
            )
        )
        return threg.dry_r(hc, hr, self._clo)

    @property
    def r_et(self):
        """r_et : numpy.ndarray (17,) w (Evaporative) heat resistances between the skin and ambience areas by local body segments [(m2*kPa)/W]."""
        hc = threg.fixed_hc(
            threg.conv_coef(
                self._posture,
                self._va,
                self._ta,
                self.t_skin,
            ),
            self._va,
        )
        return threg.wet_r(hc, self._clo, self._iclo)

    @property
    def w(self):
        """W : numpy.ndarray (17,) Skin wettedness on local body segments [-]."""
        err_cr = self.t_core - self.cr_set_point
        err_sk = self.t_skin - self.sk_set_point
        wet, *_ = threg.evaporation(
            err_cr,
            err_sk,
            self.t_skin,
            self._ta,
            self._rh,
            self.r_et,
            self._bsa_rate,
            self._age,
        )
        return wet

    @property
    def w_mean(self):
        """w_mean : float Mean skin wettedness of the whole body [-]."""
        wet = self.w
        return np.average(wet, weights=Default.local_bsa)

    @property
    def t_skin_mean(self):
        """t_skin_mean : float Mean skin temperature of the whole body [°C]."""
        return np.average(self._t_body[INDEX["skin"]], weights=Default.local_bsa)

    @property
    def t_skin(self):
        """t_skin : numpy.ndarray (17,) Skin temperatures by the local body segments [°C]."""
        return self._t_body[INDEX["skin"]].copy()

    @t_skin.setter
    def t_skin(self, inp):
        self._t_body[INDEX["skin"]] = _to17array(inp)

    @property
    def t_core(self):
        """t_core : numpy.ndarray (17,) Skin temperatures by the local body segments [°C]."""
        return self._t_body[INDEX["core"]].copy()

    @property
    def t_cb(self):
        """t_cb : numpy.ndarray (1,) Temperature at central blood pool [°C]."""
        return self._t_body[0].copy()

    @property
    def t_artery(self):
        """t_artery : numpy.ndarray (17,) Arterial temperatures by the local body segments [°C]."""
        return self._t_body[INDEX["artery"]].copy()

    @property
    def t_vein(self):
        """t_vein : numpy.ndarray (17,) Vein temperatures by the local body segments [°C]."""
        return self._t_body[INDEX["vein"]].copy()

    @property
    def t_superficial_vein(self):
        """t_superficial_vein : numpy.ndarray (12,) Superficial vein temperatures by the local body segments [°C]."""
        return self._t_body[INDEX["sfvein"]].copy()

    @property
    def t_muscle(self):
        """t_muscle : numpy.ndarray (2,) Muscle temperatures of head and pelvis [°C]."""
        return self._t_body[INDEX["muscle"]].copy()

    @property
    def t_fat(self):
        """t_fat : numpy.ndarray (2,) fat temperatures of head and pelvis  [°C]."""
        return self._t_body[INDEX["fat"]].copy()

    @property
    def body_names(self):
        """body_names : list JOS3 body names"""
        return BODY_NAMES

    @property
    def results(self):
        """Results of the model: dict."""
        return self.dict_results()

    @property
    def bmr(self):
        """Bmr : float Basal metabolic rate [W/m2]."""
        tcr = threg.basal_met(
            self._height,
            self._weight,
            self._age,
            self._sex,
            self._bmr_equation,
        )
        return tcr / self.bsa.sum()

    @property
    def version(self):
        """Version : float The current version of pythermalcomfort."""
        return self._version
