import torch
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import yaml
import platform

@dataclass
class PlanetConfig:
    """Configuration for planet-specific parameters."""
    name: str
    # Physical properties
    radius_km: float  # Planet radius in kilometers
    gravity_ms2: float  # Surface gravity in m/s²
    solar_irradiance_wm2: float  # Solar energy received at top of atmosphere
    albedo: float  # Planet's average albedo

    # Atmospheric properties
    atmospheric_mass_kg: float  # Total mass of atmosphere
    primary_greenhouse_gas: str  # Main greenhouse gas (e.g., CO2, CH4)
    greenhouse_effect_coefficient: float  # Greenhouse effect strength
    atmospheric_heat_capacity: float  # Heat capacity of atmosphere

    # Initial conditions
    initial_surface_temp_celsius: float
    initial_greenhouse_gas_ppm: float
    initial_atmospheric_pressure_kpa: float

    # Reference values (for normalization)
    reference_year: int  # Starting year for simulation
    reference_temp_celsius: float  # Reference temperature
    reference_greenhouse_gas_ppm: float  # Reference greenhouse gas concentration

    @classmethod
    def earth(cls) -> 'PlanetConfig':
        """Create Earth configuration."""
        return cls(
            name="Earth",
            radius_km=6371.0,
            gravity_ms2=9.81,
            solar_irradiance_wm2=1361.0,
            albedo=0.3,
            atmospheric_mass_kg=5.15e18,
            primary_greenhouse_gas="CO2",
            greenhouse_effect_coefficient=0.85,
            atmospheric_heat_capacity=1.0,
            initial_surface_temp_celsius=13.7,  # 1900 average
            initial_greenhouse_gas_ppm=295.7,   # 1900 CO2 level
            initial_atmospheric_pressure_kpa=101.325,
            reference_year=1900,
            reference_temp_celsius=13.7,
            reference_greenhouse_gas_ppm=280.0  # Pre-industrial CO2
        )

    @classmethod
    def venus(cls) -> 'PlanetConfig':
        """Create Venus configuration."""
        return cls(
            name="Venus",
            radius_km=6052.0,
            gravity_ms2=8.87,
            solar_irradiance_wm2=2601.3,
            albedo=0.77,
            atmospheric_mass_kg=4.8e20,
            primary_greenhouse_gas="CO2",
            greenhouse_effect_coefficient=0.99,
            atmospheric_heat_capacity=2.0,
            initial_surface_temp_celsius=462.0,
            initial_greenhouse_gas_ppm=965000.0,
            initial_atmospheric_pressure_kpa=9200.0,
            reference_year=1960,  # First Venus probe
            reference_temp_celsius=462.0,
            reference_greenhouse_gas_ppm=965000.0
        )

    @classmethod
    def mars(cls) -> 'PlanetConfig':
        """Create Mars configuration."""
        return cls(
            name="Mars",
            radius_km=3389.5,
            gravity_ms2=3.72,
            solar_irradiance_wm2=586.2,
            albedo=0.25,
            atmospheric_mass_kg=2.5e16,
            primary_greenhouse_gas="CO2",
            greenhouse_effect_coefficient=0.15,
            atmospheric_heat_capacity=0.5,
            initial_surface_temp_celsius=-63.0,
            initial_greenhouse_gas_ppm=953000.0,
            initial_atmospheric_pressure_kpa=0.636,
            reference_year=1965,  # First Mars probe
            reference_temp_celsius=-63.0,
            reference_greenhouse_gas_ppm=953000.0
        )

    @classmethod
    def titan(cls) -> 'PlanetConfig':
        """Create Titan configuration."""
        return cls(
            name="Titan",
            radius_km=2574.7,
            gravity_ms2=1.352,
            solar_irradiance_wm2=14.8,
            albedo=0.22,
            atmospheric_mass_kg=9.1e18,
            primary_greenhouse_gas="CH4",
            greenhouse_effect_coefficient=0.21,
            atmospheric_heat_capacity=1.2,
            initial_surface_temp_celsius=-179.5,
            initial_greenhouse_gas_ppm=50000.0,  # Methane
            initial_atmospheric_pressure_kpa=146.7,
            reference_year=2004,  # Cassini arrival
            reference_temp_celsius=-179.5,
            reference_greenhouse_gas_ppm=50000.0
        )

@dataclass
class SimulationConfig:
    """Configuration for the climate simulation."""
    planet: PlanetConfig
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
        self.config = config or SimulationConfig(planet=PlanetConfig.earth())
        self.device = torch.device(self.config.device)
        self.year = 0  # Track simulation year

        # Initialize state tensors
        self.temperature = torch.full(
            (self.config.grid_size, self.config.grid_size),
            self.config.planet.initial_surface_temp_celsius,
            device=self.device
        )
        self.greenhouse_gas = torch.full(
            (self.config.grid_size, self.config.grid_size),
            self.config.planet.initial_greenhouse_gas_ppm,
            device=self.device
        )
        self.pressure = torch.full(
            (self.config.grid_size, self.config.grid_size),
            self.config.planet.initial_atmospheric_pressure_kpa,
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

        # Update greenhouse gas levels
        gas_change = self._calculate_greenhouse_gas_change()
        self.greenhouse_gas += gas_change

        # Increment year
        self.year += self.config.time_step

    def _calculate_temperature_change(self) -> torch.Tensor:
        """Calculate temperature change for the current step."""
        # Constants for temperature response to greenhouse gas
        climate_sensitivity = 4.0  # °C per doubling of primary greenhouse gas
        reference_gas = self.config.planet.reference_greenhouse_gas_ppm

        # Calculate radiative forcing
        solar_input = (1 - self.config.planet.albedo) * self.config.planet.solar_irradiance_wm2 / 4
        greenhouse_forcing = climate_sensitivity * torch.log2(self.greenhouse_gas / reference_gas)

        # Calculate target temperature based on energy balance
        target_temp = (
            self.config.planet.reference_temp_celsius +
            greenhouse_forcing * self.config.planet.greenhouse_effect_coefficient
        )

        # Temperature tendency towards equilibrium
        temp_difference = target_temp - self.temperature
        relaxation_time = 15.0 * self.config.planet.atmospheric_heat_capacity

        # Temperature change from relaxation
        temp_change = temp_difference / relaxation_time

        # Add natural variability (scaled by atmospheric mass)
        variability_scale = 0.03 * (self.config.planet.atmospheric_mass_kg / 5.15e18)**0.5
        natural_variability = torch.randn_like(self.temperature) * variability_scale

        return torch.clamp(
            temp_change * self.config.time_step + natural_variability,
            min=-0.5,
            max=0.5
        )

    def _calculate_greenhouse_gas_change(self) -> torch.Tensor:
        """Calculate greenhouse gas concentration change for the current step."""
        # Natural removal rate (scaled by gravity and atmospheric mass)
        removal_rate = 0.003 * (self.config.planet.gravity_ms2 / 9.81)
        natural_removal = removal_rate * (self.greenhouse_gas - self.config.planet.reference_greenhouse_gas_ppm)

        # Base emission rate (can be modified by interventions)
        emission_rate = self._get_emission_rate()

        # Add natural variability (scaled by atmospheric mass)
        variability_scale = 0.1 * (self.config.planet.atmospheric_mass_kg / 5.15e18)**0.5
        natural_variability = torch.randn_like(self.greenhouse_gas) * variability_scale

        return (emission_rate - natural_removal + natural_variability) * self.config.time_step

    def _get_emission_rate(self) -> float:
        """Get emission rate based on planet type and simulation year."""
        if self.config.planet.name == "Earth":
            # Historical Earth emission patterns
            if self.year < 50:  # Before 1950
                return 0.4
            elif self.year < 70:  # 1950-1970
                return 1.0
            elif self.year < 90:  # 1970-1990
                return 1.5
            elif self.year < 110:  # 1990-2010
                return 2.2
            elif self.year < 120:  # 2010-2020
                return 2.5
            else:  # After 2020
                return 2.3
        else:
            # For other planets, assume constant natural emissions
            return 0.1

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
            "greenhouse_gas": self.greenhouse_gas,
            "pressure": self.pressure
        }

    def save_config(self, path: str):
        """Save the simulation configuration to a YAML file."""
        config_dict = {
            "grid_size": self.config.grid_size,
            "time_step": self.config.time_step,
            "simulation_years": self.config.simulation_years,
            "initial_temperature": self.config.planet.initial_surface_temp_celsius,
            "initial_greenhouse_gas_ppm": self.config.planet.initial_greenhouse_gas_ppm,
            "initial_atmospheric_pressure_kpa": self.config.planet.initial_atmospheric_pressure_kpa,
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