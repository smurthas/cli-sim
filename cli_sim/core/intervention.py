from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import torch
import numpy as np

@dataclass
class Intervention:
    """Class representing a climate intervention."""
    name: str
    start_year: int
    end_year: int
    intensity: float
    ramp_up_years: int
    type: str  # emission_reduction, renewable_energy, carbon_capture,
              # stratospheric_aerosol, ocean_fertilization, solar_radiation
    location: Optional[Tuple[float, float]] = None  # (lat, lon) for localized interventions

    def is_active(self, year: int) -> bool:
        """Check if the intervention is active in the given year."""
        return self.start_year <= year <= self.end_year

    def get_effect(self, year: int) -> float:
        """Calculate the effect of the intervention in the given year."""
        if not self.is_active(year):
            return 0.0

        # Calculate ramp-up effect
        if year < self.start_year + self.ramp_up_years:
            ramp_up_factor = (year - self.start_year) / self.ramp_up_years
        else:
            ramp_up_factor = 1.0

        # Calculate effect based on intervention type
        if self.type == "emission_reduction":
            return self.intensity * ramp_up_factor * 0.2  # Reduced effect to match historical data
        elif self.type == "renewable_energy":
            return self.intensity * ramp_up_factor * 0.15  # Reduced effect
        elif self.type == "carbon_capture":
            return self.intensity * ramp_up_factor * 0.1  # Reduced effect
        elif self.type == "stratospheric_aerosol":
            return self.intensity * ramp_up_factor * 0.1  # Cooling effect in °C per unit intensity
        elif self.type == "ocean_fertilization":
            return self.intensity * ramp_up_factor * 0.3  # CO2 absorption rate in ppm/year
        elif self.type == "solar_radiation":
            return self.intensity * ramp_up_factor * 0.2  # Cooling effect in °C per unit intensity
        else:
            return 0.0

    def get_spatial_effect(self, lat: float, lon: float) -> float:
        """Calculate the spatial distribution of the intervention effect."""
        # Base effect is uniform, but can be modified based on location
        base_effect = self.intensity

        # For localized interventions, apply distance-based decay
        if self.location is not None:
            lat_dist = abs(lat - self.location[0])
            lon_dist = abs(lon - self.location[1])
            distance = np.sqrt(lat_dist**2 + lon_dist**2)
            distance_factor = np.exp(-distance / 30.0)  # 30-degree decay scale
            base_effect *= distance_factor

        # Modify effect based on latitude (more effect in populated areas)
        lat_factor = 1.0 - abs(lat) / 90.0  # More effect near equator

        # Modify effect based on longitude (more effect in developed regions)
        if -120 <= lon <= -60:  # Americas
            lon_factor = 1.2
        elif -10 <= lon <= 40:  # Europe/Africa
            lon_factor = 1.1
        elif 100 <= lon <= 150:  # Asia/Pacific
            lon_factor = 1.3
        else:
            lon_factor = 0.8

        return base_effect * lat_factor * lon_factor

    def apply(self, state: Dict[str, torch.Tensor], year: int, interventions: Optional[List['Intervention']] = None) -> Dict[str, torch.Tensor]:
        """Apply the intervention effects to the current state."""
        if not self.is_active(year):
            return state

        result = state.copy()
        effect = self.get_effect(year)

        # Create spatial effect tensor
        spatial_effect = torch.tensor([
            [self.get_spatial_effect(lat, lon)
             for lon in np.linspace(-180, 180, state['temperature'].shape[1])]
            for lat in np.linspace(-90, 90, state['temperature'].shape[0])
        ])

        # Apply effects based on intervention type
        if self.type == "emission_reduction":
            # Emission reduction directly reduces CO2 growth
            base_effect = effect * spatial_effect * state['co2_ppm'] * 0.002
            if interventions and any(i.type == "carbon_capture" for i in interventions):
                base_effect *= 1.5  # 50% stronger when combined with carbon capture
            result['co2_ppm'] -= base_effect
        elif self.type == "renewable_energy":
            # Renewable energy has both direct and indirect effects
            base_temp_effect = effect * spatial_effect * 0.1
            base_co2_effect = effect * spatial_effect * state['co2_ppm'] * 0.001
            if interventions and any(i.type == "emission_reduction" for i in interventions):
                base_co2_effect *= 1.3  # 30% stronger when combined with emission reduction
            result['temperature'] -= base_temp_effect
            result['co2_ppm'] -= base_co2_effect
        elif self.type == "carbon_capture":
            # Carbon capture has a stronger effect when combined with emission reduction
            base_effect = effect * spatial_effect * state['co2_ppm'] * 0.004
            if interventions and any(i.type == "emission_reduction" for i in interventions):
                base_effect *= 1.5  # 50% stronger when combined with emission reduction
            result['co2_ppm'] -= base_effect
        elif self.type == "stratospheric_aerosol":
            result['temperature'] -= effect * spatial_effect
        elif self.type == "ocean_fertilization":
            # Ocean fertilization is more effective with higher CO2 levels
            base_effect = effect * spatial_effect
            co2_boost = torch.log2(state['co2_ppm'] / 280.0) * 0.1
            result['co2_ppm'] -= base_effect * (1.0 + co2_boost)
            result['ocean_ph'] += base_effect * 0.01
        elif self.type == "solar_radiation":
            # Solar radiation management has stronger cooling at higher temperatures
            base_effect = effect * spatial_effect
            temp_boost = torch.clamp(state['temperature'] - 13.71, min=0) * 0.1
            result['temperature'] -= base_effect * (1.0 + temp_boost)

        return result