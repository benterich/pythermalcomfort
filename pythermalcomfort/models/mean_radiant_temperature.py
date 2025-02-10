from dataclasses import dataclass
from typing import Union, List

import numpy as np
from math import pow
from math import sqrt

from pythermalcomfort.utilities import (
    units_converter,
)


#add an optional input for a weightening as an array?

# -------------------------- ------------------------- ------------------------- ------------------------- 
# MRT ESTIMATION Forth Power and linear

def mean_radiant_temperature(
    surface_temps: Union[List[float], np.ndarray],
    angle_factors: Union[List[float], np.ndarray],
    method: str = "linear",
    units: str = "SI",
) -> float:
    """Determines the adaptive thermal comfort based on EN 16798-1 2019 [3]_

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : ...

    Returns
    -------
    tmp_cmf : float, int, or array-like
        Comfort temperature at that specific running mean temperature, default in [°C]
        or in [°F]
    acceptability_cat_i : ..

    Notes
    -----
    You can use this function to calculate ..

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import mean_radiant_temperature

    """


    surface_temps = np.array(surface_temps, dtype=float)
    angle_factors = np.array(angle_factors, dtype=float)

    if not np.isclose(np.sum(angle_factors), 1.0):
        raise ValueError("Angle factors must sum to 1.")

    if units.lower() == "ip":
        surface_temps = np.array(
            units_converter(from_units="ip", **{"tmp_surface": surface_temps})
        )

    if method.lower() == "linear":
        mrt = np.sum(angle_factors * surface_temps)
    elif method.lower() == "fourthpower":
        mrt = np.sum(angle_factors * (surface_temps + 273.15) ** 4) ** 0.25 - 273.15
    else:
        raise ValueError("Unsupported method. Choose 'linear' or 'fourthpower'.")

    if units.lower() == "ip":
        mrt = units_converter(from_units="si", **{"tmp_mrt": mrt})[0]

    return mrt

# -------------------------- ------------------------- ------------------------- ------------------------- 
# MRT ESTIMATION - from simulation tools


# What do we want to have as inputs?

# Meterological Data: Solar Radiation, Air Temperature, Humidity, Wind Speed, Cloud Cover (Cannot for now), Dew Point Temperature
# Human Data: Position? Clothing (hence absorption and emissivity)

# We limit ourselves to not use a 3d model as simulation tools
# thus no sky view Factor, and assumptions regarding surrounding surfaces and their emissivities
# for now we neglect vegetation? - maybe later

# -------------------------- 
### Long Wave Radiation Sky
# -------------------------- 

#### Sky Emissivity (https://doi.org/10.26868/25222708.2017.569) "the atmospheric longwave radiation is straightforward to obtain, as long as the sky emissivity is known"

# Swinbank 1963 based on ambient temperatuer --> has shown larger deviations
# based on Water vapor pressure --> moslty used in meterology, thus i propose we use this for now - hence we can only caluclate clear sky for now and do not consider cloud cover. 
# Dilley, Prata and A˚ ngstro¨m algorithms were the best clear-sky algorithms. https://doi.org/10.1029/2008WR007394
# based on dew point temperature --> common practise in BPS software  https://doi.org/10.26868/25222708.2017.569

def calculate_sky_emissivity(water_vapor_pressure): # see comment
    """
    Calculates sky emissivity based on water vapor pressure using an empirical regression model.

    Args:
    water_vapor_pressure: Water vapor pressure in hPa.
    alpha1: Constant coefficient.
    beta1: Constant coefficient.
    gamma: Exponent.

    Returns:
    Sky emissivity (dimensionless).
    """

    alpha1 = 0.7
    beta1 = 0.003
    gamma = 0.5

    return alpha1 + beta1 * pow(water_vapor_pressure, gamma)

# Flechinger 2009 prosposed different coefficiynet for different regions: (https://doi.org/10.1029/2008WR007394) however i am nut soure which values to use, we can use this method, and just set them, greatly simplefying out approach or use a different approch, for exampel Brunt
# water vapor pressure could be optained for exmaple by August-Roche-Magnus using meterological data

def brunt_sky_emissivity(dew_point_temperature):
    """
    Calculates sky emissivity using Brunt's equation.

    Args:
    dew_point_temperature: Dew point temperature in degrees Celsius.

    Returns:
    Sky emissivity (dimensionless).
    """

    return 0.741 + 0.0062 * dew_point_temperature


#### Sky Temperature (https://doi.org/10.26868/25222708.2017.569)

def calculate_sky_temperature(air_temperature, dew_point_temperature):
    """
    Calculates the sky temperature using the given equation.

    Args:
    sky_emissivity: Emissivity of the sky (dimensionless, between 0 and 1).
    air_temperature: Air temperature in Kelvin.

    Returns:
    Sky temperature in Kelvin.
    """

    sky_emissivity = brunt_sky_emissivity(dew_point_temperature)

    return sky_emissivity ** 0.25 * air_temperature


# ------> 

def sky_long_wave_radiation(dew_point_temperature, air_temperature):
    """
    Calculates the long-wave radiation emitted by the sky.

    Args:
    sky_emissivity: Emissivity of the sky (dimensionless, between 0 and 1).
    sky_temperature: Temperature of the sky in Kelvin.

    Returns:
    Long-wave radiation emitted by the sky (W/m²).
    """

    stefan_boltzmann_constant = 5.67e-8 
    sky_temperature = calculate_sky_temperature(air_temperature, dew_point_temperature)
    sky_emissivity = brunt_sky_emissivity(dew_point_temperature)

    return sky_emissivity * stefan_boltzmann_constant * pow(sky_temperature, 4)


# -------------------------- 
### Long Wave Radiation Ground and Urban Faces
# -------------------------- 


# ground emsisivity typicall 0.95, depending on the surface, we could provide this as an optional input
# ground temperature, simplist is to assume ground temperature == to Air Temperture?, this can vary by multiple degrees, https://doi.org/10.1016/0002-1571(81)90105-9, https://doi.org/10.1130/GES02448.1
# or just ground tempateure as a input, since its often available in weather data, but at lower levels
# ground temperature can be drawn with some emmpirical relations to air temp and solar rad, but are location specific - maybe we can incooperate this later? Havent found much good regarding this yet.

def estimate_soil_temperature(air_temperature, solar_radiation, k=0.07):
    """
    Estimate soil surface temperature based on air temperature and solar radiation.

    Args:
        air_temperature (float): Air temperature in Kelvin.
        solar_radiation (float): Incoming solar radiation (W/m²).
        k (float): Empirical coefficient (default=0.07). Adjusts for soil heat capacity and albedo. Usuallz between 0.02 and 0.1?

    Returns:
        float: Estimated soil surface temperature in Kelvin.
    """
    
    return air_temperature + k * solar_radiation

# Example Usage:
air_temp = 300  # Air temperature in Kelvin
solar_rad = 800  # Solar radiation in W/m²
soil_temp = estimate_soil_temperature(air_temp, solar_rad)
print(f"Estimated Soil Temperature: {soil_temp:.2f} K")


def ground_long_wave_radiation(air_temperature, solar_rad):
    """
    Calculates the long-wave radiation emitted by the ground.

    Args:
    ground_emissivity: Emissivity of the ground (dimensionless, typically around 0.9).
    ground_temperature: Temperature of the ground in Kelvin.

    Returns:
    Long-wave radiation emitted by the ground (W/m²).
    """
    #ground_temperature = air_temperature ## assumption for now or optional input
    ground_temperature = estimate_soil_temperature(air_temperature, solar_rad)
    ground_emissivity = 0.95 #optional input later
    stefan_boltzmann_constant = 5.67e-8  # W/m²K^4

    return ground_emissivity * stefan_boltzmann_constant * pow(ground_temperature, 4)

def calculate_surface_longwave_radiation(emissivity, temperature):
    """
    Calculates the longwave radiation emitted by a surface.

    Args:
        emissivity: Emissivity of the surface (dimensionless).
        temperature: Temperature of the surface in Kelvin.

    Returns:
        Longwave radiation emitted by the surface (W/m²).
    """

    stefan_boltzmann_constant = 5.67e-8  # W/m²K^4

    return emissivity * stefan_boltzmann_constant * pow(temperature, 4)


# -------------------------- 
### Person
# -------------------------- 

# Maybe estimate absorption coefficient based on clothing factor 


# -------------------------- 
### MRT
# -------------------------- 


# http://dx.doi.org/10.1016/j.buildenv.2014.05.019 from CityComfort+ based on ASHREA 2009 handbook or
# 10.1088/1742-6596/2069/1/012186



def calculate_mrt(
    ak, fp, direct_solar_irradiance,
    al, surface_view_factors, surface_emissivities, surface_temperatures,
    dew_point_temperature, air_temperature, solar_radiation,
    Fground_p, ground_emissivity=0.95,
    stefan_boltzmann_constant=5.67e-8
):
    """
    Calculates MRT including longwave radiation from ground and sky.

    Args:
        ak: Absorption coefficient for shortwave radiation. 
        fp: Projected area factor of the human body. 
        direct_solar_irradiance: Direct solar irradiance (W/m²).
        al: Absorption coefficient for longwave radiation.                          (STILL MISSING)
        surface_view_factors: View factors between each surface and the person.
        surface_emissivities: Emissivities of the surfaces.
        surface_temperatures: Temperatures of the surfaces (Kelvin).
        dew_point_temperature: Dew point temperature (°C).
        air_temperature: Air temperature (Kelvin).
        solar_radiation: Solar radiation (W/m²).
        Fsky_p: Sky view factor.
        ground_emissivity: Emissivity of the ground surface. Default is 0.95.
        stefan_boltzmann_constant: Stefan-Boltzmann constant. Default is 5.67e-8.

    Returns:
        MRT in Kelvin.
    """
    
    # ak: For a clothed human body, akak​ is typically in the range of 0.6 to 0.8, exposed skin 0.7, maybe we can just estime it based on an input clothing factor
    # fp: simplefied to cube and cylinder, to ease view factor caluclation, about 0.5 for a standing human. 0.2 to 0.3 sitting, etc.

    # Shortwave absorbed by the person
    shortwave_flux = ak * fp * direct_solar_irradiance

    # Longwave radiation from the sky
    sky_radiation = sky_long_wave_radiation(dew_point_temperature, air_temperature)

    # Longwave radiation from the ground
    ground_radiation = ground_long_wave_radiation(air_temperature, solar_radiation, ground_emissivity)

    # Longwave radiation from surrounding surfaces
    surface_radiations = [
        calculate_surface_longwave_radiation(emissivity, temperature)
        for emissivity, temperature in zip(surface_emissivities, surface_temperatures)
    ]
    surface_radiations.append(ground_radiation)  # Include ground radiation
    surface_view_factors.append(Fground_p)  # Adjust ground view factor

    #surface view factors need to add up to 1

    longwave_flux_surfaces = sum(
        vf * rad for vf, rad in zip(surface_view_factors, surface_radiations)
    )

    total_flux = shortwave_flux + sky_radiation + al * longwave_flux_surfaces

    return sqrt(total_flux / stefan_boltzmann_constant)

# Example Usage
# mrt = calculate_mrt_with_longwave(
#     ak=0.7, fp=0.5, direct_solar_irradiance=900,
#     al=0.95, surface_view_factors=[0.2, 0.3, 0.2],
#     surface_emissivities=[0.9, 0.8, 0.7], surface_temperatures=[305, 300, 295],
#     dew_point_temperature=10, air_temperature=300, solar_radiation=800,
#     Fsky_p=0.5
# )


# Questions?
# Is an skyView Factor input feasable and useful? Or too complicated? What are the alternatives?
# There are multiple ways to estimate ground longwave radiation and sky_long_wave_radiation, can we be able to input methods?
# same if even nessary for persons caluclations


# FROM OLD
# def calculate_mrt(
#     surface_emissivities, surface_temperatures,
#     Fground_p, ground_emissivity=0.95,
# )
    

# View Factor Example List: Urban Canyon (3 different ones?), and more maybe computed from grassopper?

# MRT Setup?
def calculate_mrt(MrtDefaults, sky_radiation, ground_radiation, direct_solar_irradiance, SurfaceViewFactor, SurfaceRadiation):
    """
    MrtParams: Class holding specific values
    sky_radiation: Method result for long wave radiation form the sky
    ground_radiation: Method result long wave radiation from the ground
    direct_solar_irradiance:
    SurfaceViewFactor: Method result, including ground?

    """

    stefan_boltzmann_constant=5.67e-8

    return 0


class MrtDefaults(Enum):
    ak: float = 0.7 
    fp:  float = 0.7 
    al: float = 0.7


#MrtDefaults.ak.value


#dataclass input with touple for the vew factors and tempertuares, think about how to provide this