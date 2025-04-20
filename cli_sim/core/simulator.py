import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import yaml
import platform
from datetime import datetime
from .intervention import Intervention
import os

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
    """Configuration for climate simulation."""
    grid_size: int
    time_step: float
    simulation_years: int
    initial_temperature: float
    initial_co2_ppm: float
    initial_ocean_ph: float
    heat_capacity: float
    albedo: float
    greenhouse_effect: float

class ClimateSimulator:
    """Simulates climate change over time with interventions."""

    def __init__(self, config):
        self.grid_size = config.grid_size
        self.time_step = config.time_step
        self.simulation_years = config.simulation_years
        self.year = 1900  # Start year for historical simulation

        # Initialize state tensors
        self.temperature = torch.full((self.grid_size, self.grid_size), config.initial_temperature)
        self.co2 = torch.full((self.grid_size, self.grid_size), config.initial_co2_ppm)
        self.ocean_ph = torch.full((self.grid_size, self.grid_size), config.initial_ocean_ph)

        # Initialize intervention effects
        self.intervention_effects = {
            'temperature': torch.zeros((self.grid_size, self.grid_size)),
            'co2': torch.zeros((self.grid_size, self.grid_size)),
            'ocean_ph': torch.zeros((self.grid_size, self.grid_size))
        }

        # Create latitude and longitude grids
        self.lat_grid = torch.linspace(-90, 90, self.grid_size).view(-1, 1).expand(self.grid_size, self.grid_size)
        self.lon_grid = torch.linspace(-180, 180, self.grid_size).view(1, -1).expand(self.grid_size, self.grid_size)

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current simulation state."""
        return {
            'temperature': self.temperature,
            'co2_ppm': self.co2,
            'ocean_ph': self.ocean_ph
        }

    def _calculate_temperature_change(self, interventions: List[Intervention]) -> torch.Tensor:
        """Calculate temperature change based on CO2 and interventions."""
        # Base temperature change from CO2 using improved climate sensitivity model
        co2_change = self.co2 - 280.0  # Change from pre-industrial

        # Updated climate sensitivity with thermal inertia and feedback effects
        base_sensitivity = 3.0  # Base climate sensitivity
        feedback_factor = 1.0 + torch.log2(self.co2 / 280.0) * 0.1  # Increased sensitivity at higher CO2
        climate_sensitivity = base_sensitivity * feedback_factor

        # Ocean thermal inertia varies with depth
        surface_inertia = 0.2
        deep_ocean_factor = 0.05
        thermal_inertia = surface_inertia + deep_ocean_factor * torch.log2(self.co2 / 280.0)

        # Calculate temperature change with thermal inertia
        target_temp_change = climate_sensitivity * torch.log2(self.co2 / 280.0)
        current_temp_change = self.temperature - 13.71  # Difference from 1900 baseline

        # Temperature change moves toward target with thermal inertia
        temp_change = (target_temp_change - current_temp_change) * thermal_inertia

        # Add intervention effects
        state = self.get_state()
        for intervention in interventions:
            if intervention.is_active(self.year):
                state = intervention.apply(state, self.year, interventions)

        # Add natural variability with reduced magnitude at higher temperatures
        base_variability = 0.03
        damping_factor = torch.exp(-current_temp_change / 2.0)
        natural_variability = torch.randn_like(temp_change) * base_variability * damping_factor

        return (state['temperature'] - self.temperature) + temp_change + natural_variability

    def _calculate_co2_change(self, interventions: List[Intervention]) -> torch.Tensor:
        """Calculate CO2 change based on emissions and interventions."""
        # Base CO2 change from historical emissions with feedback effects
        if self.year < 1950:
            base_growth = 0.003  # ~0.3% annual growth
        elif self.year < 1970:
            base_growth = 0.007  # ~0.7% annual growth
        elif self.year < 1990:
            base_growth = 0.012  # ~1.2% annual growth
        elif self.year < 2010:
            base_growth = 0.018  # ~1.8% annual growth
        elif self.year < 2020:
            base_growth = 0.020  # ~2.0% annual growth
        else:
            base_growth = 0.015  # ~1.5% projected growth

        # Add temperature feedback on natural carbon sinks
        temp_effect = torch.clamp(self.temperature - 13.71, min=0) * 0.001
        co2_growth = base_growth + temp_effect

        # Calculate base CO2 change from emissions
        co2_change = self.co2 * co2_growth

        # Natural carbon sinks with saturation
        base_removal = 0.0015  # Base removal rate (0.15% per year)
        sink_saturation = torch.exp(-(self.co2 - 280.0) / 800.0)  # Decreasing efficiency at higher CO2
        natural_removal = self.co2 * base_removal * sink_saturation

        # Net change before interventions
        net_change = co2_change - natural_removal

        # Add intervention effects
        state = self.get_state()
        for intervention in interventions:
            if intervention.is_active(self.year):
                new_state = intervention.apply(state, self.year, interventions)
                # Only apply CO2 changes from interventions
                net_change = net_change + (new_state['co2_ppm'] - state['co2_ppm'])
                state = new_state

        return net_change

    def _calculate_ph_change(self, interventions: List[Intervention]) -> torch.Tensor:
        """Calculate ocean pH change based on CO2 concentration."""
        # pH change based on CO2 with chemical buffering and temperature effects
        co2_change = self.co2 - 280.0  # Change from pre-industrial

        # Base pH sensitivity with temperature dependence
        base_sensitivity = 0.001  # pH units per ppm CO2
        temp_factor = 1.0 + torch.clamp(self.temperature - 13.71, min=0) * 0.02

        # Chemical buffering (decreases with higher CO2 and temperature)
        buffer_capacity = torch.exp(-co2_change / 700.0) * torch.exp(-(self.temperature - 13.71) / 10.0)
        effective_sensitivity = base_sensitivity * temp_factor * buffer_capacity

        ph_change = co2_change * effective_sensitivity

        # Add intervention effects
        state = self.get_state()
        for intervention in interventions:
            if intervention.is_active(self.year):
                state = intervention.apply(state, self.year, interventions)

        # Reduced natural variability
        natural_variability = torch.randn_like(ph_change) * 0.0003
        return (state['ocean_ph'] - self.ocean_ph) + natural_variability

    def step(self, interventions: Optional[List[Intervention]] = None) -> None:
        """Advance simulation by one time step."""
        if interventions is None:
            interventions = []

        # Calculate changes
        temp_change = self._calculate_temperature_change(interventions)
        co2_change = self._calculate_co2_change(interventions)
        ph_change = self._calculate_ph_change(interventions)

        # Update state
        self.temperature += temp_change * self.time_step
        self.co2 += co2_change * self.time_step
        self.ocean_ph += ph_change * self.time_step

        # Update year
        self.year += 1

    def run_simulation(self, years: np.ndarray, interventions: List[Intervention]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run simulation for given years with interventions."""
        temps = []
        co2s = []
        phs = []

        for year in years:
            self.step(interventions)
            temps.append(self.temperature.mean().item())
            co2s.append(self.co2.mean().item())
            phs.append(self.ocean_ph.mean().item())

        return np.array(temps), np.array(co2s), np.array(phs)

    def save_config(self, path: str):
        """Save the simulation configuration to a YAML file."""
        # Create output directory with timestamp if not provided
        if not os.path.isabs(path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("outputs", f"simulation_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "config.yaml")

        config_dict = {
            "grid_size": self.grid_size,
            "time_step": self.time_step,
            "simulation_years": self.simulation_years,
            "initial_temperature": self.temperature.mean().item(),
            "initial_co2_ppm": self.co2.mean().item(),
            "initial_ocean_ph": self.ocean_ph.mean().item(),
            "interventions": [intervention.__dict__ for intervention in interventions]
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
            intervention_obj = Intervention(**intervention)
            simulator.add_intervention(intervention_obj)

        return simulator