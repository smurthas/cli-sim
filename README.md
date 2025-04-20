# Climate Simulation System

A Python-based climate simulation system that models global climate dynamics, including temperature changes, CO2 concentrations, and ocean pH levels.

## Features

- Grid-based climate simulation
- Historical data validation
- Interactive visualization
- Customizable interventions with synergistic effects
- Comprehensive validation suite

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Simulations

1. Historical Simulation:
```bash
python -m cli_sim.examples.historical_simulation
```

2. Custom Simulation:
```python
from cli_sim.core.simulator import ClimateSimulator, SimulationConfig
from cli_sim.core.intervention import Intervention

# Configure simulation
config = SimulationConfig(
    grid_size=64,
    time_step=0.25,
    simulation_years=50,
    initial_temperature=15.0,
    initial_co2_ppm=400.0,
    initial_ocean_ph=8.1,
    heat_capacity=4.184,
    albedo=0.3,
    greenhouse_effect=0.95
)

# Create simulator
simulator = ClimateSimulator(config)

# Add interventions with synergistic effects
interventions = [
    Intervention(
        name="Emission Reduction",
        start_year=0,
        end_year=50,
        intensity=0.5,
        ramp_up_years=10,
        type="emission_reduction"
    ),
    Intervention(
        name="Carbon Capture",
        start_year=0,
        end_year=50,
        intensity=0.3,
        ramp_up_years=5,
        type="carbon_capture"
    )
]

# Run simulation
for year in range(50):
    simulator.step(interventions)
```

### Intervention Types and Synergies

The system supports several types of interventions with built-in synergistic effects:

1. Emission Reduction
   - Directly reduces CO2 growth
   - 50% stronger when combined with carbon capture
   - 30% stronger when combined with renewable energy

2. Renewable Energy
   - Reduces both temperature and CO2
   - Synergizes with emission reduction

3. Carbon Capture
   - Removes CO2 from the atmosphere
   - More effective at higher CO2 levels
   - 50% stronger when combined with emission reduction

4. Ocean Fertilization
   - Enhances natural carbon sinks
   - More effective at higher CO2 levels
   - Improves ocean pH

5. Solar Radiation Management
   - Direct cooling effect
   - More effective at higher temperatures

### Validation and Testing

The system includes a comprehensive validation suite that checks:
- Unit tests for core functionality
- Historical simulation accuracy
- Visualization output generation

To run the full validation suite:
```bash
python -m cli_sim.tests.validate_all
```

The validation script will:
1. Run all unit tests
2. Execute the historical simulation and validate its results against thresholds
3. Check that all required visualization outputs are generated
4. Provide a detailed report of any failures

Current validation thresholds and results:
- Historical Period (1900-2020):
  - Temperature RMSE: 0.33°C (threshold: 1.0°C)
  - CO2 RMSE: 19.95 ppm (threshold: 100.0 ppm)
  - Ocean pH RMSE: 0.10 (threshold: 0.1 pH units)
- Projected Period (2025-2050):
  - Temperature RMSE: 1.48°C (threshold: 3.0°C)
  - CO2 RMSE: 48.04 ppm (threshold: 250.0 ppm)
  - Ocean pH RMSE: 0.25 (threshold: 0.3 pH units)

## Output Structure

All simulation outputs are organized in the `outputs` directory with the following structure:
```
outputs/
└── historical_simulation_YYYYMMDD_HHMMSS/
    ├── historical_simulation.png
    ├── temperature_change.png
    ├── co2_change.png
    ├── ph_change.png
    └── simulation_data.json
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
