import torch
import numpy as np
from cli_sim.core.simulator import ClimateSimulator, SimulationConfig
from cli_sim.visualization.visualizer import ClimateVisualizer
from cli_sim.interventions.interventions import InterventionManager
from cli_sim.core.output_manager import OutputManager
import matplotlib.pyplot as plt
from datetime import datetime
import math

def temperature_change(co2_concentration, base_temp=13.7):
    """Calculate temperature change based on CO2 concentration"""
    # Updated climate sensitivity to better match historical data
    climate_sensitivity = 3.0  # °C per doubling of CO2
    co2_pre_industrial = 280.0  # Pre-industrial CO2 level
    temp_change = climate_sensitivity * math.log2(co2_concentration / co2_pre_industrial)
    return base_temp + temp_change

def co2_change(year, current_co2):
    """Calculate CO2 change based on historical emissions patterns"""
    # Updated emissions factors to better match historical data
    if year < 1950:
        return current_co2 * (1 + 0.001)  # ~0.1% annual growth
    elif year < 1970:
        return current_co2 * (1 + 0.003)  # ~0.3% annual growth
    elif year < 1990:
        return current_co2 * (1 + 0.005)  # ~0.5% annual growth
    elif year < 2010:
        return current_co2 * (1 + 0.007)  # ~0.7% annual growth
    elif year < 2020:
        return current_co2 * (1 + 0.008)  # ~0.8% annual growth
    else:
        # Projected emissions based on IPCC SSP2-4.5 scenario
        return current_co2 * (1 + 0.006)  # ~0.6% annual growth

def ocean_ph_change(co2_concentration):
    """Calculate ocean pH based on CO2 concentration"""
    # Updated pH model based on empirical data
    base_ph = 8.25  # Pre-industrial pH
    ph_sensitivity = 0.002  # pH change per 10 ppm CO2
    co2_change = co2_concentration - 280.0  # Change from pre-industrial
    ph_change = (co2_change / 10.0) * ph_sensitivity
    return base_ph - ph_change

def run_historical_simulation():
    """Run historical climate simulation from 1900 to 2050"""
    print("Starting historical simulation (1900-2050)...")

    # Initial conditions for 1900
    years = list(range(1900, 2051))
    global_temperatures = [13.7]  # Starting temperature in 1900
    global_co2 = [295.7]  # Starting CO2 in 1900
    global_ocean_ph = [8.2]  # Starting ocean pH in 1900

    # Historical data points for validation
    historical_temps = {
        1900: 13.7,
        1950: 13.8,
        1970: 13.9,
        1990: 14.2,
        2000: 14.8,
        2005: 15.0,
        2010: 15.3,
        2015: 15.6,
        2020: 15.9
    }

    historical_co2 = {
        1900: 295.7,
        1950: 310.0,
        1970: 325.0,
        1990: 350.0,
        2000: 369.5,
        2005: 379.8,
        2010: 389.9,
        2015: 400.8,
        2020: 412.4
    }

    historical_ph = {
        1900: 8.18,
        1950: 8.16,
        1970: 8.15,
        1990: 8.13,
        2000: 8.11,
        2005: 8.10,
        2010: 8.09,
        2015: 8.08,
        2020: 8.07
    }

    # Projected values for 2025-2050 (IPCC SSP2-4.5 scenario)
    projected_temps = {
        2025: 16.1,
        2030: 16.3,
        2035: 16.5,
        2040: 16.7,
        2045: 16.9,
        2050: 17.1
    }

    projected_co2 = {
        2025: 425.0,
        2030: 435.0,
        2035: 445.0,
        2040: 455.0,
        2045: 465.0,
        2050: 475.0
    }

    projected_ph = {
        2025: 8.06,
        2030: 8.05,
        2035: 8.04,
        2040: 8.03,
        2045: 8.02,
        2050: 8.01
    }

    # Run simulation
    for year in years[1:]:  # Start from 1901
        # Calculate changes
        new_co2 = co2_change(year, global_co2[-1])
        new_temp = temperature_change(new_co2)
        new_ph = ocean_ph_change(new_co2)

        # Store results
        global_co2.append(new_co2)
        global_temperatures.append(new_temp)
        global_ocean_ph.append(new_ph)

        # Print yearly results
        print(f"Year {year}:")
        print(f"  Global Temperature: {new_temp:.2f}°C")
        print(f"  Global CO2: {new_co2:.2f} ppm")
        print(f"  Global Ocean pH: {new_ph:.2f}")

    print("\nHistorical simulation complete!")

    # Validation metrics
    print("\nValidation Metrics:")
    print("Historical Period (1900-2020):\n")

    for year in sorted(historical_temps.keys()):
        sim_idx = year - 1900
        print(f"Year {year}:")
        print(f"  Temperature - Simulated: {global_temperatures[sim_idx]:.2f}°C, Historical: {historical_temps[year]:.2f}°C")
        print(f"  CO2 - Simulated: {global_co2[sim_idx]:.2f} ppm, Historical: {historical_co2[year]:.2f} ppm")
        print(f"  pH - Simulated: {global_ocean_ph[sim_idx]:.2f}, Historical: {historical_ph[year]:.2f}\n")

    print("Projected Period (2025-2050):\n")
    for year in sorted(projected_temps.keys()):
        sim_idx = year - 1900
        print(f"Year {year}:")
        print(f"  Temperature - Simulated: {global_temperatures[sim_idx]:.2f}°C, Projected: {projected_temps[year]:.2f}°C")
        print(f"  CO2 - Simulated: {global_co2[sim_idx]:.2f} ppm, Projected: {projected_co2[year]:.2f} ppm")
        print(f"  pH - Simulated: {global_ocean_ph[sim_idx]:.2f}, Projected: {projected_ph[year]:.2f}\n")

if __name__ == "__main__":
    run_historical_simulation()