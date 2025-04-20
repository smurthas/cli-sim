import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Optional, List
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class ClimateVisualizer:
    def __init__(self, simulator):
        """Initialize the climate visualizer."""
        self.simulator = simulator
        self.fig = None
        self.ax = None

    def plot_global_map(self, variable: str, title: Optional[str] = None):
        """Create a global map visualization of a climate variable."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(15, 8))
            self.ax = plt.axes(projection=ccrs.PlateCarree())

        # Get the data
        data = self.simulator.get_state()[variable].cpu().numpy()

        # Create the plot
        self.ax.clear()
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Create the heatmap
        im = self.ax.imshow(
            data,
            transform=ccrs.PlateCarree(),
            origin='lower',
            extent=[-180, 180, -90, 90],
            cmap='viridis'
        )

        # Add colorbar
        plt.colorbar(im, ax=self.ax, label=variable)

        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f'Global {variable} Distribution')

        return self.fig

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