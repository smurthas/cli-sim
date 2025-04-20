import torch
import numpy as np
from cli_sim.core.simulator import ClimateSimulator, SimulationConfig
from cli_sim.visualization.visualizer import ClimateVisualizer
from cli_sim.interventions.interventions import InterventionManager
from cli_sim.core.output_manager import OutputManager
import matplotlib.pyplot as plt
from datetime import datetime

def run_historical_simulation():
    # Historical data for 2000
    initial_conditions = {
        'temperature': 14.8,  # Global average temperature in 2000 (°C)
        'co2_ppm': 369.52,   # CO2 concentration in 2000 (ppm)
        'ocean_ph': 8.11     # Ocean pH in 2000
    }

    # Create configuration for historical simulation
    config = SimulationConfig(
        grid_size=180,  # 2-degree resolution for faster computation
        time_step=0.25,  # Quarterly updates
        simulation_years=25,  # 2000-2025
        initial_temperature=initial_conditions['temperature'],
        initial_co2_ppm=initial_conditions['co2_ppm'],
        initial_ocean_ph=initial_conditions['ocean_ph'],
        greenhouse_effect=0.65,  # Adjusted greenhouse sensitivity
        heat_capacity=1.0,  # Normalized heat capacity
        albedo=0.3  # Earth's average albedo
    )

    # Initialize the simulator
    simulator = ClimateSimulator(config)

    # Create the intervention manager
    intervention_manager = InterventionManager()

    # Add historical interventions and events

    # Pre-2000 baseline emissions
    intervention_manager.add_intervention(
        'carbon_sequestration',
        {
            'start_time': 0.0,
            'duration': 25.0,
            'intensity': -2.0,  # Negative intensity represents emissions
            'location': None  # Global effect
        }
    )

    # Kyoto Protocol implementation (2005)
    intervention_manager.add_intervention(
        'carbon_sequestration',
        {
            'start_time': 5.0,  # 2005
            'duration': 7.0,    # Until 2012
            'intensity': 0.3,
            'location': None  # Global effect
        }
    )

    # Paris Agreement implementation (2016)
    intervention_manager.add_intervention(
        'carbon_sequestration',
        {
            'start_time': 16.0,  # 2016
            'duration': 9.0,     # Until 2025
            'intensity': 0.5,
            'location': None  # Global effect
        }
    )

    # Renewable energy expansion (gradual from 2000)
    intervention_manager.add_intervention(
        'solar_radiation_management',  # Representing reduced fossil fuel use
        {
            'start_time': 0.0,
            'duration': 25.0,
            'intensity': 0.1
        }
    )

    # Initialize the output manager
    output_manager = OutputManager("historical_simulation")

    # Initialize the visualizer
    visualizer = ClimateVisualizer(simulator, output_manager)

    # Create arrays to store historical data
    time_points = []
    global_temperatures = []
    global_co2_levels = []
    global_ph_levels = []

    # Historical validation data
    historical_temps = {
        2000: 14.8,
        2005: 15.0,
        2010: 15.3,
        2015: 15.6,
        2020: 15.9
    }

    historical_co2 = {
        2000: 369.52,
        2005: 379.80,
        2010: 389.85,
        2015: 400.83,
        2020: 412.44
    }

    historical_ph = {
        2000: 8.11,
        2005: 8.10,
        2010: 8.09,
        2015: 8.08,
        2020: 8.07
    }

    # Run the simulation
    print("Starting historical simulation (2000-2025)...")
    for step in range(int(config.simulation_years / config.time_step)):
        current_time = step * config.time_step
        year = 2000 + current_time
        time_points.append(year)

        # Get current state
        state = simulator.get_state()

        # Apply interventions
        state = intervention_manager.apply_interventions(state, current_time)

        # Update simulator state
        simulator.temperature = state['temperature']
        simulator.co2_ppm = state['co2_ppm']
        simulator.ocean_ph = state['ocean_ph']

        # Step the simulation
        simulator.step()

        # Store global averages
        global_temperatures.append(simulator.temperature.mean().item())
        global_co2_levels.append(simulator.co2_ppm.mean().item())
        global_ph_levels.append(simulator.ocean_ph.mean().item())

        # Visualize every year
        if step % 4 == 0:  # Since time_step is 0.25 (quarterly)
            print(f"Year {year:.1f}:")
            print(f"  Global Temperature: {global_temperatures[-1]:.2f}°C")
            print(f"  Global CO2: {global_co2_levels[-1]:.2f} ppm")
            print(f"  Global Ocean pH: {global_ph_levels[-1]:.2f}")

            # Create and save visualizations
            fig_temp = visualizer.plot_global_map(
                'temperature',
                f'Global Temperature in {year:.0f}'
            )
            fig_temp.savefig(output_manager.get_path(f'historical_temperature_{year:.0f}.png'))

            fig_co2 = visualizer.plot_global_map(
                'co2_ppm',
                f'Global CO2 Concentration in {year:.0f}'
            )
            fig_co2.savefig(output_manager.get_path(f'historical_co2_{year:.0f}.png'))

    print("Historical simulation complete!")

    # Plot time series of global variables with historical data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Temperature plot
    ax1.plot(time_points, global_temperatures, label='Simulation')
    ax1.scatter(historical_temps.keys(), historical_temps.values(),
                color='red', label='Historical Data')
    ax1.set_title('Global Temperature Over Time (2000-2025)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True)
    ax1.legend()

    # CO2 plot
    ax2.plot(time_points, global_co2_levels, label='Simulation')
    ax2.scatter(historical_co2.keys(), historical_co2.values(),
                color='red', label='Historical Data')
    ax2.set_title('Global CO2 Concentration Over Time (2000-2025)')
    ax2.set_ylabel('CO2 (ppm)')
    ax2.grid(True)
    ax2.legend()

    # pH plot
    ax3.plot(time_points, global_ph_levels, label='Simulation')
    ax3.scatter(historical_ph.keys(), historical_ph.values(),
                color='red', label='Historical Data')
    ax3.set_title('Global Ocean pH Over Time (2000-2025)')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('pH')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_manager.get_path('historical_global_trends.png'))

    # Plot intervention effects
    fig_interventions = visualizer.plot_intervention_effects()
    if fig_interventions:
        fig_interventions.savefig(output_manager.get_path('historical_intervention_effects.png'))

    print(f"Simulation outputs saved to: {output_manager.get_directory()}")

if __name__ == '__main__':
    run_historical_simulation()