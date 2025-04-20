import torch
import numpy as np
from cli_sim.core.simulator import ClimateSimulator, SimulationConfig
from cli_sim.visualization.visualizer import ClimateVisualizer
from cli_sim.core.intervention import Intervention
from cli_sim.core.output_manager import OutputManager

def run_historical_simulation():
    """Run a historical climate simulation from 1900 to 2050."""
    print("Starting historical simulation (1900-2050)...")

    # Historical data points for validation
    historical_temps = {
        1900: 13.71, 1910: 13.73, 1920: 13.78, 1930: 13.84,
        1940: 13.91, 1950: 13.98, 1960: 14.08, 1970: 14.21,
        1980: 14.37, 1990: 14.58, 2000: 14.85, 2010: 15.12,
        2020: 15.45
    }

    historical_co2 = {
        1900: 295.7, 1910: 299.8, 1920: 303.9, 1930: 307.9,
        1940: 311.3, 1950: 315.9, 1960: 322.3, 1970: 331.1,
        1980: 342.8, 1990: 358.4, 2000: 369.5, 2010: 388.7,
        2020: 412.5
    }

    historical_ph = {
        1900: 8.21, 1910: 8.20, 1920: 8.19, 1930: 8.18,
        1940: 8.17, 1950: 8.16, 1960: 8.15, 1970: 8.13,
        1980: 8.11, 1990: 8.09, 2000: 8.07, 2010: 8.05,
        2020: 8.03
    }

    # Initialize simulator with historical configuration
    config = SimulationConfig(
        grid_size=64,  # Increased resolution
        time_step=0.25,  # Smaller time step for better accuracy
        simulation_years=151,  # 1900-2050
        initial_temperature=13.71,  # 1900 global mean temperature
        initial_co2_ppm=295.7,  # 1900 CO2 concentration
        initial_ocean_ph=8.21,  # 1900 ocean pH
        heat_capacity=4.184,  # Water's specific heat capacity in J/g°C
        albedo=0.3,  # Earth's average albedo
        greenhouse_effect=0.95  # Strong greenhouse effect
    )

    simulator = ClimateSimulator(config)
    output_manager = OutputManager()
    visualizer = ClimateVisualizer(simulator, output_manager)

    # Define historical interventions
    interventions = [
        # Pre-1950: Early industrialization
        Intervention(
            name="Early Industrial Era",
            start_year=1900,
            end_year=1950,
            intensity=0.3,
            ramp_up_years=10,
            type="emission_reduction",
            location=(45, 0)  # Centered on Europe/North America
        ),
        # 1950-1970: Post-war boom
        Intervention(
            name="Post-war Industrial Boom",
            start_year=1950,
            end_year=1970,
            intensity=0.6,
            ramp_up_years=5,
            type="emission_reduction",
            location=(40, -100)  # Centered on North America
        ),
        # 1970-1990: Global industrialization
        Intervention(
            name="Global Industrialization",
            start_year=1970,
            end_year=1990,
            intensity=0.8,
            ramp_up_years=5,
            type="emission_reduction",
            location=(30, 100)  # Shifted towards Asia
        ),
        # 1990-2020: Modern era
        Intervention(
            name="Modern Era",
            start_year=1990,
            end_year=2020,
            intensity=1.0,
            ramp_up_years=5,
            type="emission_reduction",
            location=(30, 120)  # Centered on East Asia
        ),
        # 2020-2050: Future projection
        Intervention(
            name="Future Projection",
            start_year=2020,
            end_year=2050,
            intensity=0.7,  # Reduced intensity due to climate policies
            ramp_up_years=10,
            type="emission_reduction",
            location=(0, 0)  # Global effect
        )
    ]

    # Run simulation
    temps = []
    co2s = []
    phs = []

    for year in range(1900, 2051):
        simulator.step(interventions)
        temps.append(simulator.temperature.mean().item())
        co2s.append(simulator.co2.mean().item())
        phs.append(simulator.ocean_ph.mean().item())

    print("\nHistorical Simulation Results:")
    print("-----------------------------")
    print("Historical Period (1900-2020):")

    # Calculate validation metrics for historical period
    historical_years = list(historical_temps.keys())
    simulated_temps = [temps[year - 1900] for year in historical_years]
    simulated_co2 = [co2s[year - 1900] for year in historical_years]
    simulated_ph = [phs[year - 1900] for year in historical_years]

    temp_rmse = np.sqrt(np.mean((np.array(simulated_temps) - np.array(list(historical_temps.values())))**2))
    co2_rmse = np.sqrt(np.mean((np.array(simulated_co2) - np.array(list(historical_co2.values())))**2))
    ph_rmse = np.sqrt(np.mean((np.array(simulated_ph) - np.array(list(historical_ph.values())))**2))

    print(f"Temperature RMSE: {temp_rmse:.2f}°C")
    print(f"CO2 RMSE: {co2_rmse:.2f} ppm")
    print(f"Ocean pH RMSE: {ph_rmse:.3f}")

    print("\nProjected Period (2025-2050):")
    projected_temps = {
        2025: 15.85, 2030: 16.10, 2035: 16.35,
        2040: 16.60, 2045: 16.85, 2050: 17.10
    }

    projected_co2 = {
        2025: 425.8, 2030: 435.2, 2035: 444.7,
        2040: 454.3, 2045: 464.0, 2050: 473.8
    }

    projected_ph = {
        2025: 8.01, 2030: 7.99, 2035: 7.97,
        2040: 7.95, 2045: 7.93, 2050: 7.91
    }

    projected_years = list(projected_temps.keys())
    simulated_future_temps = [temps[year - 1900] for year in projected_years]
    simulated_future_co2 = [co2s[year - 1900] for year in projected_years]
    simulated_future_ph = [phs[year - 1900] for year in projected_years]

    temp_rmse_future = np.sqrt(np.mean((np.array(simulated_future_temps) - np.array(list(projected_temps.values())))**2))
    co2_rmse_future = np.sqrt(np.mean((np.array(simulated_future_co2) - np.array(list(projected_co2.values())))**2))
    ph_rmse_future = np.sqrt(np.mean((np.array(simulated_future_ph) - np.array(list(projected_ph.values())))**2))

    print(f"Temperature RMSE: {temp_rmse_future:.2f}°C")
    print(f"CO2 RMSE: {co2_rmse_future:.2f} ppm")
    print(f"Ocean pH RMSE: {ph_rmse_future:.3f}")

    # Use visualizer to create plots
    output_dir = visualizer.plot_historical_simulation(
        years=list(range(1900, 2051)),
        temperatures=temps,
        co2_levels=co2s,
        ph_levels=phs,
        historical_temps=historical_temps,
        historical_co2=historical_co2,
        historical_ph=historical_ph,
        projected_temps=projected_temps,
        projected_co2=projected_co2,
        projected_ph=projected_ph
    )

    print(f"\nSimulation outputs saved to: {output_dir}")

if __name__ == "__main__":
    run_historical_simulation()