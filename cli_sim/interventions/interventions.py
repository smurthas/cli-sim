import torch
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class InterventionParameters:
    """Base class for intervention parameters."""
    start_time: float
    duration: float
    intensity: float
    location: Optional[tuple] = None  # (lat, lon) for localized interventions

class StratosphericAerosolInjection:
    """Model for stratospheric aerosol injection intervention."""
    def __init__(self, parameters: InterventionParameters):
        self.params = parameters

    def apply(self, temperature: torch.Tensor, time: float) -> torch.Tensor:
        """Apply the cooling effect of stratospheric aerosols."""
        if not (self.params.start_time <= time <= self.params.start_time + self.params.duration):
            return torch.zeros_like(temperature)

        # Simple cooling effect model
        cooling_factor = self.params.intensity * 0.1  # °C per unit intensity
        return -cooling_factor * torch.ones_like(temperature)

class CarbonSequestration:
    """Model for carbon sequestration intervention."""
    def __init__(self, parameters: InterventionParameters):
        self.params = parameters

    def apply(self, co2_ppm: torch.Tensor, time: float) -> torch.Tensor:
        """Apply the CO2 removal effect of carbon sequestration."""
        if not (self.params.start_time <= time <= self.params.start_time + self.params.duration):
            return torch.zeros_like(co2_ppm)

        # CO2 removal rate in ppm/year
        removal_rate = self.params.intensity * 0.5
        return -removal_rate * torch.ones_like(co2_ppm)

class OceanFertilization:
    """Model for ocean fertilization intervention."""
    def __init__(self, parameters: InterventionParameters):
        self.params = parameters

    def apply(self, co2_ppm: torch.Tensor, ocean_ph: torch.Tensor, time: float) -> Dict[str, torch.Tensor]:
        """Apply the effects of ocean fertilization."""
        if not (self.params.start_time <= time <= self.params.start_time + self.params.duration):
            return {
                'co2_change': torch.zeros_like(co2_ppm),
                'ph_change': torch.zeros_like(ocean_ph)
            }

        # CO2 absorption and pH change rates
        co2_absorption = self.params.intensity * 0.3  # ppm/year
        ph_increase = self.params.intensity * 0.01  # pH units/year

        return {
            'co2_change': -co2_absorption * torch.ones_like(co2_ppm),
            'ph_change': ph_increase * torch.ones_like(ocean_ph)
        }

class SolarRadiationManagement:
    """Model for solar radiation management intervention."""
    def __init__(self, parameters: InterventionParameters):
        self.params = parameters

    def apply(self, temperature: torch.Tensor, time: float) -> torch.Tensor:
        """Apply the cooling effect of solar radiation management."""
        if not (self.params.start_time <= time <= self.params.start_time + self.params.duration):
            return torch.zeros_like(temperature)

        # Cooling effect based on reduced solar radiation
        cooling_factor = self.params.intensity * 0.2  # °C per unit intensity
        return -cooling_factor * torch.ones_like(temperature)

class InterventionManager:
    """Manager class for handling multiple climate interventions."""
    def __init__(self):
        self.interventions = []

    def add_intervention(self, intervention_type: str, parameters: Dict):
        """Add a new intervention to the manager."""
        params = InterventionParameters(**parameters)

        if intervention_type == 'stratospheric_aerosol_injection':
            intervention = StratosphericAerosolInjection(params)
        elif intervention_type == 'carbon_sequestration':
            intervention = CarbonSequestration(params)
        elif intervention_type == 'ocean_fertilization':
            intervention = OceanFertilization(params)
        elif intervention_type == 'solar_radiation_management':
            intervention = SolarRadiationManagement(params)
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

        self.interventions.append(intervention)

    def apply_interventions(self, state: Dict[str, torch.Tensor], time: float) -> Dict[str, torch.Tensor]:
        """Apply all active interventions to the current state."""
        result = state.copy()

        for intervention in self.interventions:
            if isinstance(intervention, StratosphericAerosolInjection):
                result['temperature'] += intervention.apply(state['temperature'], time)
            elif isinstance(intervention, CarbonSequestration):
                result['co2_ppm'] += intervention.apply(state['co2_ppm'], time)
            elif isinstance(intervention, OceanFertilization):
                effects = intervention.apply(state['co2_ppm'], state['ocean_ph'], time)
                result['co2_ppm'] += effects['co2_change']
                result['ocean_ph'] += effects['ph_change']
            elif isinstance(intervention, SolarRadiationManagement):
                result['temperature'] += intervention.apply(state['temperature'], time)

        return result