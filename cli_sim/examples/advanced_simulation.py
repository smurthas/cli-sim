import torch
import numpy as np
from cli_sim.core.simulator import ClimateSimulator, SimulationConfig
from cli_sim.visualization.visualizer import ClimateVisualizer
from cli_sim.interventions.interventions import InterventionManager

def run_advanced_simulation():
    # Create a high-resolution configuration
    config = SimulationConfig(
        grid_size=360,  # 1-degree resolution
        time_step=0.05,  # 0.05 years per step
        simulation_years=100,
        initial_temperature=15.0,
        initial_co2_ppm=450.0,  # Higher initial CO2
        initial_ocean_ph=8.0    # Slightly more acidic
    )

    # Initialize the simulator
    simulator = ClimateSimulator(config)

    # Create the intervention manager
    intervention_manager = InterventionManager()

    # Add a sequence of coordinated interventions
    # 1. Initial carbon sequestration
    intervention_manager.add_intervention(
        'carbon_sequestration',
        {
            'start_time': 5.0,
            'duration': 30.0,
            'intensity': 1.5,
            'location': (45, -90)  # North America
        }
    )

    # 2. Ocean fertilization
    intervention_manager.add_intervention(
        'ocean_fertilization',
        {
            'start_time': 10.0,
            'duration': 20.0,
            'intensity': 0.8,
            'location': (0, -180)  # Pacific Ocean
        }
    )

    # 3. Stratospheric aerosol injection
    intervention_manager.add_intervention(
        'stratospheric_aerosol_injection',
        {
            'start_time': 15.0,
            'duration': 15.0,
            'intensity': 0.7
        }
    )

    # 4. Additional carbon sequestration
    intervention_manager.add_intervention(
        'carbon_sequestration',
        {
            'start_time': 25.0,
            'duration': 25.0,
            'intensity': 2.0,
            'location': (-30, 150)  # Australia
        }
    )

    # 5. Solar radiation management
    intervention_manager.add_intervention(
        'solar_radiation_management',
        {
            'start_time': 30.0,
            'duration': 20.0,
            'intensity': 0.6
        }
    )

    # Initialize the visualizer
    visualizer = ClimateVisualizer(simulator)

    # Create arrays to store historical data
    time_points = []
    global_temperatures = []
    global_co2_levels = []
    global_ph_levels = []

    # Run the simulation
    print("Starting advanced simulation...")
    for step in range(int(config.simulation_years / config.time_step)):
        current_time = step * config.time_step
        time_points.append(current_time)

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

        # Visualize every 20 steps
        if step % 20 == 0:
            print(f"Year {current_time:.1f}:")
            print(f"  Global Temperature: {global_temperatures[-1]:.2f}°C")
            print(f"  Global CO2: {global_co2_levels[-1]:.2f} ppm")
            print(f"  Global Ocean pH: {global_ph_levels[-1]:.2f}")

            # Create and save visualizations
            fig_temp = visualizer.plot_global_map(
                'temperature',
                f'Global Temperature at year {current_time:.1f}'
            )
            fig_temp.savefig(f'advanced_temperature_{step}.png')

            fig_co2 = visualizer.plot_global_map(
                'co2_ppm',
                f'Global CO2 Concentration at year {current_time:.1f}'
            )
            fig_co2.savefig(f'advanced_co2_{step}.png')

            # Create interactive plot with all variables
            fig_interactive = visualizer.create_interactive_plot(
                ['temperature', 'co2_ppm', 'ocean_ph']
            )
            fig_interactive.write_html(f'advanced_climate_state_{step}.html')

    print("Advanced simulation complete!")

    # Plot intervention effects
    fig_interventions = visualizer.plot_intervention_effects()
    if fig_interventions:
        fig_interventions.savefig('advanced_intervention_effects.png')

    # Plot time series of global variables
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Temperature plot
    ax1.plot(time_points, global_temperatures)
    ax1.set_title('Global Temperature Over Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True)

    # CO2 plot
    ax2.plot(time_points, global_co2_levels)
    ax2.set_title('Global CO2 Concentration Over Time')
    ax2.set_ylabel('CO2 (ppm)')
    ax2.grid(True)

    # pH plot
    ax3.plot(time_points, global_ph_levels)
    ax3.set_title('Global Ocean pH Over Time')
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('pH')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('advanced_global_trends.png')

if __name__ == '__main__':
    run_advanced_simulation()