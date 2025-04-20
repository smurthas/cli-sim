import torch
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import yaml
import platform

@dataclass
class SimulationConfig:
    """Configuration for the climate simulation."""
    grid_size: int = 360  # 1-degree resolution
    time_step: float = 1.0  # years
    simulation_years: int = 100
    device: str = "mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else "cpu"

    # Initial conditions
    initial_temperature: float = 15.0  # Celsius
    initial_co2_ppm: float = 400.0
    initial_ocean_ph: float = 8.1

    # Model parameters
    heat_capacity: float = 1.0
    albedo: float = 0.3
    greenhouse_effect: float = 0.8

class ClimateSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the climate simulator."""
        self.config = config or SimulationConfig()
        self.device = torch.device(self.config.device)

        # Initialize state tensors
        self.temperature = torch.full(
            (self.config.grid_size, self.config.grid_size),
            self.config.initial_temperature,
            device=self.device
        )
        self.co2_ppm = torch.full(
            (self.config.grid_size, self.config.grid_size),
            self.config.initial_co2_ppm,
            device=self.device
        )
        self.ocean_ph = torch.full(
            (self.config.grid_size, self.config.grid_size),
            self.config.initial_ocean_ph,
            device=self.device
        )

        # Initialize intervention effects
        self.interventions: List[Dict] = []

    def add_intervention(self, intervention_type: str, parameters: Dict):
        """Add a climate intervention to the simulation."""
        self.interventions.append({
            "type": intervention_type,
            "parameters": parameters,
            "start_time": len(self.interventions)  # Simple sequential timing
        })

    def step(self):
        """Perform one simulation step."""
        # Update temperature based on current state and interventions
        temp_change = self._calculate_temperature_change()
        self.temperature += temp_change

        # Update CO2 levels
        co2_change = self._calculate_co2_change()
        self.co2_ppm += co2_change

        # Update ocean pH
        ph_change = self._calculate_ph_change()
        self.ocean_ph += ph_change

    def _calculate_temperature_change(self) -> torch.Tensor:
        """Calculate temperature change for the current step."""
        # Constants
        solar_constant = 1361.0  # W/m²
        earth_albedo = 0.3
        greenhouse_base = 0.33  # Adjusted for Earth's natural greenhouse effect
        stefan_boltzmann = 5.67e-8  # W/m²/K⁴
        seconds_per_year = 365.25 * 24 * 3600

        # Reference temperature and CO2 levels (year 2000)
        ref_temp = 14.8  # °C
        ref_co2 = 369.52  # ppm

        # Temperature response to CO2 changes (climate sensitivity)
        # ~3°C per doubling of CO2
        co2_sensitivity = 3.0 * torch.log2(self.co2_ppm / ref_co2)

        # Calculate temperature tendency towards equilibrium
        temp_difference = self.temperature - (ref_temp + co2_sensitivity)
        relaxation_time = 5.0  # years

        # Temperature change from relaxation
        temp_change = -temp_difference / relaxation_time

        # Add natural variability (reduced magnitude)
        natural_variability = torch.randn_like(self.temperature) * 0.01

        return torch.clamp(
            temp_change * self.config.time_step + natural_variability,
            min=-0.5,
            max=0.5
        )

    def _calculate_co2_change(self) -> torch.Tensor:
        """Calculate CO2 concentration change for the current step."""
        # Basic carbon cycle model
        natural_removal = 0.01 * self.co2_ppm
        human_emissions = 2.0 * (1.0 + 0.02 * (self.config.simulation_years / 100.0))  # Increasing emissions

        # Apply interventions
        intervention_effect = torch.zeros_like(self.co2_ppm)
        for intervention in self.interventions:
            if intervention["type"] == "carbon_sequestration":
                intervention_effect -= intervention["parameters"]["rate"]

        # Add natural variability
        natural_variability = torch.randn_like(self.co2_ppm) * 0.1

        return (human_emissions - natural_removal + intervention_effect + natural_variability) * self.config.time_step

    def _calculate_ph_change(self) -> torch.Tensor:
        """Calculate ocean pH change for the current step."""
        # Basic ocean acidification model
        co2_effect = -0.001 * (self.co2_ppm - 400.0)

        # Add natural variability
        natural_variability = torch.randn_like(self.ocean_ph) * 0.01

        return (co2_effect + natural_variability) * self.config.time_step

    def run(self, steps: Optional[int] = None):
        """Run the simulation for the specified number of steps."""
        if steps is None:
            steps = int(self.config.simulation_years / self.config.time_step)

        for _ in range(steps):
            self.step()

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get the current state of the simulation."""
        return {
            "temperature": self.temperature,
            "co2_ppm": self.co2_ppm,
            "ocean_ph": self.ocean_ph
        }

    def save_config(self, path: str):
        """Save the simulation configuration to a YAML file."""
        config_dict = {
            "grid_size": self.config.grid_size,
            "time_step": self.config.time_step,
            "simulation_years": self.config.simulation_years,
            "initial_temperature": self.config.initial_temperature,
            "initial_co2_ppm": self.config.initial_co2_ppm,
            "initial_ocean_ph": self.config.initial_ocean_ph,
            "interventions": self.interventions
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f)

    @classmethod
    def load_config(cls, path: str) -> 'ClimateSimulator':
        """Load a simulation configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = SimulationConfig(**{
            k: v for k, v in config_dict.items()
            if k in SimulationConfig.__dataclass_fields__
        })

        simulator = cls(config)
        for intervention in config_dict.get("interventions", []):
            simulator.add_intervention(
                intervention["type"],
                intervention["parameters"]
            )

        return simulator