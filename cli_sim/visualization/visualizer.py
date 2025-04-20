import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Optional, List, Tuple, Union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pathlib import Path
from cli_sim.core.output_manager import OutputManager
from cli_sim.core.simulator import PlanetConfig

class ClimateVisualizer:
    def __init__(self, simulator, output_manager: OutputManager):
        """Initialize the climate visualizer."""
        self.simulator = simulator
        self.output_manager = output_manager
        self.fig = None
        self.ax = None

    def plot_global_map(self, variable: str, title: Optional[str] = None):
        """Create a global map visualization of a climate variable."""
        # Create a new figure for each plot
        fig = plt.figure(figsize=(15, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Get the data
        data = self.simulator.get_state()[variable].cpu().numpy()

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Create the heatmap with appropriate colormap and scale
        if variable == 'temperature':
            cmap = 'RdYlBu_r'
            label = 'Temperature (°C)'
        elif variable == 'co2_ppm':
            cmap = 'YlOrBr'
            label = 'CO2 (ppm)'
        elif variable == 'ocean_ph':
            cmap = 'viridis'
            label = 'Ocean pH'
        else:
            cmap = 'viridis'
            label = variable

        # Create the heatmap
        im = ax.imshow(
            data,
            transform=ccrs.PlateCarree(),
            origin='lower',
            extent=[-180, 180, -90, 90],
            cmap=cmap
        )

        # Add colorbar with appropriate label
        plt.colorbar(im, ax=ax, label=label)

        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f'Global {variable} Distribution')

        return fig

    def plot_validation_results(self, df: pd.DataFrame, historical_data: Dict, planet_config: PlanetConfig) -> Path:
        """Plot simulation results against historical data and return the output path."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Temperature plot
        ax1.plot(df['year'], df['temperature'], label='Simulated', color='blue')
        if isinstance(historical_data['temperature'], dict):  # Earth-style data
            years = list(historical_data['temperature'].keys())
            temps = list(historical_data['temperature'].values())
            ax1.scatter(years, temps, color='red', label='Historical', alpha=0.6)
        else:  # Other planets with uncertainty
            years, temps, uncertainties = zip(*historical_data['temperature'])
            ax1.errorbar(years, temps, yerr=uncertainties, fmt='o', color='red',
                        label='Observed', alpha=0.6, capsize=5)

        ax1.set_xlabel('Year')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title(f'{planet_config.name} Surface Temperature')
        ax1.grid(True)
        ax1.legend()

        # Greenhouse gas plot
        ax2.plot(df['year'], df['greenhouse_gas'], label='Simulated', color='blue')
        if isinstance(historical_data['greenhouse_gas'], dict):  # Earth-style data
            years = list(historical_data['greenhouse_gas'].keys())
            gases = list(historical_data['greenhouse_gas'].values())
            ax2.scatter(years, gases, color='red', label='Historical', alpha=0.6)
        else:  # Other planets with uncertainty
            years, gases, uncertainties = zip(*historical_data['greenhouse_gas'])
            ax2.errorbar(years, gases, yerr=uncertainties, fmt='o', color='red',
                        label='Observed', alpha=0.6, capsize=5)

        ax2.set_xlabel('Year')
        ax2.set_ylabel(f'{planet_config.primary_greenhouse_gas} (ppm)')
        ax2.set_title(f'{planet_config.name} {planet_config.primary_greenhouse_gas} Concentration')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        # Save the plot using the output manager
        output_path = self.output_manager.get_output_path(
            f"{planet_config.name.lower()}_validation.png"
        )
        plt.savefig(output_path)
        plt.close()

        return output_path

    def create_interactive_plot(self, variables: Optional[List[str]] = None):
        """Create an interactive plot using Plotly."""
        if variables is None:
            variables = ['temperature', 'co2_ppm', 'ocean_ph']

        state = self.simulator.get_state()

        # Create figure
        fig = go.Figure()

        # Add traces for each variable
        for var in variables:
            data = state[var].cpu().numpy()
            fig.add_trace(go.Heatmap(
                z=data,
                colorscale='Viridis',
                name=var,
                showscale=True
            ))

        # Update layout
        fig.update_layout(
            title='Climate Simulation Variables',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            height=800,
            width=1200
        )

        return fig

    def plot_time_series(self, variable: str, location: tuple = (0, 0)):
        """Plot the time series of a variable at a specific location."""
        # This would require storing historical data
        # For now, we'll just plot the current state
        data = self.simulator.get_state()[variable].cpu().numpy()
        value = data[location[0], location[1]]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([0], [value], 'o-')
        ax.set_xlabel('Time')
        ax.set_ylabel(variable)
        ax.set_title(f'{variable} at location {location}')

        return fig

    def save_animation(self, variable: str, filename: str, fps: int = 10):
        """Save an animation of the simulation over time."""
        # This would require storing historical data
        # Implementation would use matplotlib.animation
        pass

    def plot_intervention_effects(self):
        """Plot the effects of climate interventions."""
        interventions = self.simulator.interventions
        if not interventions:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot intervention effects over time
        for intervention in interventions:
            ax.axvline(x=intervention['start_time'],
                      color='red',
                      linestyle='--',
                      label=f"{intervention['type']} start")

        ax.set_xlabel('Time')
        ax.set_ylabel('Effect')
        ax.set_title('Climate Intervention Timeline')
        ax.legend()

        return fig