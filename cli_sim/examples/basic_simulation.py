import torch
from cli_sim.core.simulator import ClimateSimulator, SimulationConfig
from cli_sim.visualization.visualizer import ClimateVisualizer
from cli_sim.interventions.interventions import InterventionManager

def run_basic_simulation():
    # Create a custom configuration
    config = SimulationConfig(
        grid_size=180,  # 2-degree resolution
        time_step=0.1,  # 0.1 years per step
        simulation_years=50,
        initial_temperature=15.0,
        initial_co2_ppm=420.0,
        initial_ocean_ph=8.1
    )

    # Initialize the simulator
    simulator = ClimateSimulator(config)

    # Create the intervention manager
    intervention_manager = InterventionManager()

    # Add some climate interventions
    intervention_manager.add_intervention(
        'carbon_sequestration',
        {
            'start_time': 10.0,
            'duration': 20.0,
            'intensity': 1.0
        }
    )

    intervention_manager.add_intervention(
        'stratospheric_aerosol_injection',
        {
            'start_time': 15.0,
            'duration': 10.0,
            'intensity': 0.5
        }
    )

    # Initialize the visualizer
    visualizer = ClimateVisualizer(simulator)

    # Run the simulation
    print("Starting simulation...")
    for step in range(int(config.simulation_years / config.time_step)):
        # Get current state
        state = simulator.get_state()

        # Apply interventions
        state = intervention_manager.apply_interventions(state, step * config.time_step)

        # Update simulator state
        simulator.temperature = state['temperature']
        simulator.co2_ppm = state['co2_ppm']
        simulator.ocean_ph = state['ocean_ph']

        # Step the simulation
        simulator.step()

        # Visualize every 10 steps
        if step % 10 == 0:
            # Create and save visualizations
            fig_temp = visualizer.plot_global_map('temperature', f'Temperature at year {step * config.time_step:.1f}')
            fig_temp.savefig(f'temperature_{step}.png')

            fig_co2 = visualizer.plot_global_map('co2_ppm', f'CO2 Concentration at year {step * config.time_step:.1f}')
            fig_co2.savefig(f'co2_{step}.png')

            # Create interactive plot
            fig_interactive = visualizer.create_interactive_plot()
            fig_interactive.write_html(f'climate_state_{step}.html')

    print("Simulation complete!")

    # Plot intervention effects
    fig_interventions = visualizer.plot_intervention_effects()
    if fig_interventions:
        fig_interventions.savefig('intervention_effects.png')

if __name__ == '__main__':
    run_basic_simulation()