import numpy as np
from cli_sim.core.simulator import ClimateSimulator, SimulationConfig, PlanetConfig
from cli_sim.core.output_manager import OutputManager
from cli_sim.visualization.visualizer import ClimateVisualizer
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

class PlanetaryData:
    """Historical and observational data for planets."""

    @staticmethod
    def earth_data() -> Dict[str, Dict[int, float]]:
        """Historical Earth data."""
        return {
            "temperature": {
                1900: 13.7, 1950: 13.8, 1970: 13.9, 1990: 14.2,
                2000: 14.8, 2005: 15.0, 2010: 15.3, 2015: 15.6,
                2020: 15.9
            },
            "greenhouse_gas": {  # CO2 in ppm
                1900: 295.7, 1950: 310.0, 1970: 325.0, 1990: 350.0,
                2000: 369.5, 2005: 379.8, 2010: 389.9, 2015: 400.8,
                2020: 412.4
            }
        }

    @staticmethod
    def venus_data() -> Dict[str, List[Tuple[int, float, float]]]:
        """Venus observational data (year, value, uncertainty)."""
        return {
            "temperature": [
                (1962, 462.0, 2.0),  # Mariner 2
                (1974, 464.0, 1.5),  # Venera 9
                (1978, 462.0, 1.0),  # Pioneer Venus
                (1990, 463.0, 1.0),  # Magellan
                (2006, 462.0, 1.0),  # Venus Express
            ],
            "greenhouse_gas": [  # CO2 in ppm
                (1962, 965000.0, 5000.0),
                (1974, 964000.0, 4000.0),
                (1978, 965000.0, 3000.0),
                (1990, 965000.0, 2000.0),
                (2006, 965000.0, 2000.0),
            ]
        }

    @staticmethod
    def mars_data() -> Dict[str, List[Tuple[int, float, float]]]:
        """Mars observational data (year, value, uncertainty)."""
        return {
            "temperature": [
                (1965, -63.0, 5.0),  # Mariner 4
                (1976, -63.0, 3.0),  # Viking 1
                (1997, -63.0, 2.0),  # Mars Pathfinder
                (2004, -63.0, 1.0),  # Mars Express
                (2012, -62.0, 1.0),  # Curiosity
            ],
            "greenhouse_gas": [  # CO2 in ppm
                (1965, 953000.0, 10000.0),
                (1976, 953000.0, 5000.0),
                (1997, 953000.0, 3000.0),
                (2004, 953000.0, 2000.0),
                (2012, 953000.0, 2000.0),
            ]
        }

    @staticmethod
    def titan_data() -> Dict[str, List[Tuple[int, float, float]]]:
        """Titan observational data (year, value, uncertainty)."""
        return {
            "temperature": [
                (1980, -179.5, 3.0),  # Voyager 1
                (2004, -179.5, 1.0),  # Cassini arrival
                (2005, -179.2, 0.5),  # Huygens probe
                (2010, -179.4, 0.5),  # Cassini extended
                (2017, -179.5, 0.5),  # Cassini final
            ],
            "greenhouse_gas": [  # CH4 in ppm
                (1980, 50000.0, 5000.0),
                (2004, 50000.0, 2000.0),
                (2005, 49800.0, 1000.0),
                (2010, 50000.0, 1000.0),
                (2017, 50000.0, 1000.0),
            ]
        }

def validate_planet(planet_config: PlanetConfig, historical_data: Dict, output_manager: OutputManager) -> Tuple[pd.DataFrame, Dict[str, float], Path]:
    """Run simulation and validate against historical data for a planet."""
    # Configure simulation
    config = SimulationConfig(
        planet=planet_config,
        grid_size=180,  # 2-degree resolution
        time_step=1.0,
        simulation_years=150  # Long enough for all planets
    )

    # Initialize simulator and visualizer
    simulator = ClimateSimulator(config)
    visualizer = ClimateVisualizer(simulator, output_manager)

    # Storage for results
    results = {
        'year': [],
        'temperature': [],
        'greenhouse_gas': []
    }

    # Run simulation
    start_year = planet_config.reference_year
    for year in range(start_year, start_year + config.simulation_years):
        state = simulator.get_state()
        results['year'].append(year)
        results['temperature'].append(float(state['temperature'].mean()))
        results['greenhouse_gas'].append(float(state['greenhouse_gas'].mean()))
        simulator.step()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Calculate validation metrics
    metrics = {}
    if isinstance(historical_data['temperature'], dict):  # Earth-style data
        temp_errors = []
        gas_errors = []
        for year, temp in historical_data['temperature'].items():
            if year in df['year'].values:
                sim_temp = df[df['year'] == year]['temperature'].iloc[0]
                temp_errors.append((sim_temp - temp)**2)
        for year, gas in historical_data['greenhouse_gas'].items():
            if year in df['year'].values:
                sim_gas = df[df['year'] == year]['greenhouse_gas'].iloc[0]
                gas_errors.append((sim_gas - gas)**2)

        metrics['temperature_rmse'] = np.sqrt(np.mean(temp_errors))
        metrics['greenhouse_gas_rmse'] = np.sqrt(np.mean(gas_errors))
    else:  # Other planets with uncertainty data
        temp_errors = []
        gas_errors = []
        for year, temp, _ in historical_data['temperature']:
            if year in df['year'].values:
                sim_temp = df[df['year'] == year]['temperature'].iloc[0]
                temp_errors.append((sim_temp - temp)**2)
        for year, gas, _ in historical_data['greenhouse_gas']:
            if year in df['year'].values:
                sim_gas = df[df['year'] == year]['greenhouse_gas'].iloc[0]
                gas_errors.append((sim_gas - gas)**2)

        metrics['temperature_rmse'] = np.sqrt(np.mean(temp_errors))
        metrics['greenhouse_gas_rmse'] = np.sqrt(np.mean(gas_errors))

    # Generate validation plot
    plot_path = visualizer.plot_validation_results(df, historical_data, planet_config)

    return df, metrics, plot_path

def run_planetary_validation():
    """Run validation for all planets."""
    # Initialize output manager
    output_manager = OutputManager()
    output_dir = output_manager.get_output_directory()
    print(f"Output directory: {output_dir}")

    planets = {
        'Earth': (PlanetConfig.earth(), PlanetaryData.earth_data()),
        'Venus': (PlanetConfig.venus(), PlanetaryData.venus_data()),
        'Mars': (PlanetConfig.mars(), PlanetaryData.mars_data()),
        'Titan': (PlanetConfig.titan(), PlanetaryData.titan_data())
    }

    print("Running planetary validation...")
    for planet_name, (planet_config, historical_data) in planets.items():
        print(f"\nValidating {planet_name}...")
        df, metrics, plot_path = validate_planet(planet_config, historical_data, output_manager)

        print(f"Results for {planet_name}:")
        print(f"  Temperature RMSE: {metrics['temperature_rmse']:.2f}°C")
        print(f"  {planet_config.primary_greenhouse_gas} RMSE: {metrics['greenhouse_gas_rmse']:.2f} ppm")
        print(f"  Validation plot saved to: {plot_path}")

    print(f"\nValidation complete! All plots have been saved to: {output_dir}")

if __name__ == "__main__":
    run_planetary_validation()