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

    # Create configuration for 1900 conditions
    config = SimulationConfig(
        grid_size=180,  # 2-degree resolution
        time_step=1.0,  # 1 year per step
        simulation_years=151,  # 1900 to 2050
        initial_temperature=13.7,  # 1900 global average temperature
        initial_co2_ppm=295.7,  # 1900 CO2 concentration
        initial_ocean_ph=8.2,  # 1900 ocean pH
        heat_capacity=1.0,
        albedo=0.3,
        greenhouse_effect=0.85  # Adjusted for historical accuracy
    )

    # Initialize components
    simulator = ClimateSimulator(config)
    intervention_manager = InterventionManager()
    output_manager = OutputManager("historical_simulation")

    # Add historical interventions
    intervention_manager.add_intervention(
        'carbon_sequestration',
        {
            'start_time': 50,  # 1950s: Beginning of major reforestation efforts
            'duration': 70,
            'intensity': 0.2
        }
    )

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

    # Store simulation results
    years = list(range(1900, 2051))
    global_temperatures = []
    global_co2 = []
    global_ocean_ph = []

    # Run simulation
    print("\nRunning simulation year by year...")
    for year in years:
        # Get current state
        state = simulator.get_state()

        # Store global averages
        global_temperatures.append(float(state['temperature'].mean()))
        global_co2.append(float(state['co2_ppm'].mean()))
        global_ocean_ph.append(float(state['ocean_ph'].mean()))

        # Print yearly results
        print(f"Year {year}:")
        print(f"  Global Temperature: {global_temperatures[-1]:.2f}°C")
        print(f"  Global CO2: {global_co2[-1]:.2f} ppm")
        print(f"  Global Ocean pH: {global_ocean_ph[-1]:.2f}")

        # Apply interventions and step simulation
        state = intervention_manager.apply_interventions(state, year - 1900)
        simulator.temperature = state['temperature']
        simulator.co2_ppm = state['co2_ppm']
        simulator.ocean_ph = state['ocean_ph']
        simulator.step()

    print("\nHistorical simulation complete!")

    # Create visualization of global trends
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Temperature plot
    ax1.plot(years, global_temperatures, label='Simulated', color='blue')
    for year, temp in historical_temps.items():
        ax1.plot(year, temp, 'ro', label='Historical' if year == 1900 else None)
    for year, temp in projected_temps.items():
        ax1.plot(year, temp, 'go', label='Projected' if year == 2025 else None)
    ax1.set_title('Global Temperature Over Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True)
    ax1.legend()

    # CO2 plot
    ax2.plot(years, global_co2, label='Simulated', color='blue')
    for year, co2 in historical_co2.items():
        ax2.plot(year, co2, 'ro', label='Historical' if year == 1900 else None)
    for year, co2 in projected_co2.items():
        ax2.plot(year, co2, 'go', label='Projected' if year == 2025 else None)
    ax2.set_title('Global CO2 Concentration Over Time')
    ax2.set_ylabel('CO2 (ppm)')
    ax2.grid(True)
    ax2.legend()

    # pH plot
    ax3.plot(years, global_ocean_ph, label='Simulated', color='blue')
    for year, ph in historical_ph.items():
        ax3.plot(year, ph, 'ro', label='Historical' if year == 1900 else None)
    for year, ph in projected_ph.items():
        ax3.plot(year, ph, 'go', label='Projected' if year == 2025 else None)
    ax3.set_title('Global Ocean pH Over Time')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('pH')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_manager.get_path('historical_global_trends.png'))

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