# Planet Climate Simulator

A GPU-accelerated climate simulation system built with PyTorch that models the complex interactions between Earth's major systems: biosphere, pedosphere, geosphere, hydrosphere, and atmosphere.

## Features

- GPU-accelerated climate modeling using PyTorch
- Interactive simulation with real-time visualization
- Configurable initial conditions and parameters
- Support for modeling climate intervention technologies:
  - Stratospheric aerosol injection
  - Carbon sequestration
  - Ocean fertilization
  - Solar radiation management
  - And more
- Beautiful visualizations of climate variables and system interactions
- Modular architecture for easy extension

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from climate_simulator import ClimateSimulator

# Initialize simulator with default parameters
simulator = ClimateSimulator()

# Run simulation
simulator.run()

# Visualize results
simulator.visualize()
```

## Project Structure

- `cli_sim/` - Main package directory
  - `core/` - Core simulation components
  - `models/` - PyTorch models for different Earth systems
  - `visualization/` - Visualization tools
  - `config/` - Configuration management
  - `interventions/` - Climate intervention implementations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU
- Additional dependencies listed in requirements.txt

## License

MIT License
