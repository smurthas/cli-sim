"""
Climate Visualization Module

This module contains all visualization-related code for the climate simulation.
All plotting and visualization functions should be implemented here to maintain
a clean separation of concerns and ensure consistent visualization behavior.

Key principles:
1. All visualization code should be in this module
2. Each visualization function should be a method of the ClimateVisualizer class
3. Output files should be saved through the OutputManager
4. Visualization functions should be reusable across different simulation types
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Optional, List, Tuple, Union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import json
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

    def plot_intervention_effects(self, show_timeline: bool = True) -> Union[plt.Figure, Tuple[plt.Figure, plt.Figure]]:
        """Plot the effects of climate interventions with detailed impact analysis."""
        interventions = self.simulator.interventions
        if not interventions:
            return None

        # Create figure for intervention timeline and intensities
        fig_timeline = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 1, height_ratios=[1, 2])

        # Timeline subplot
        ax_timeline = fig_timeline.add_subplot(gs[0])
        ax_effects = fig_timeline.add_subplot(gs[1])

        # Plot intervention timeline
        colors = {'emission_reduction': 'green', 'renewable_energy': 'blue', 'carbon_capture': 'purple'}
        y_positions = {}
        current_pos = 0

        for intervention in interventions:
            int_type = intervention['type']
            if int_type not in y_positions:
                y_positions[int_type] = current_pos
                current_pos += 1

            start = intervention['start_time']
            end = intervention['end_time']
            y_pos = y_positions[int_type]

            # Plot intervention period
            ax_timeline.fill_between([start, end], [y_pos-0.3]*2, [y_pos+0.3]*2,
                                  color=colors.get(int_type, 'gray'), alpha=0.3)

            # Add ramp-up indication
            ramp_end = start + intervention['ramp_up_time']
            ax_timeline.fill_between([start, ramp_end], [y_pos-0.3]*2, [y_pos+0.3]*2,
                                  color=colors.get(int_type, 'gray'), alpha=0.15, hatch='/')

        ax_timeline.set_yticks(list(y_positions.values()))
        ax_timeline.set_yticklabels(list(y_positions.keys()))
        ax_timeline.set_title('Intervention Timeline')
        ax_timeline.grid(True, alpha=0.3)

        # Plot cumulative effects
        years = range(
            min(int['start_time'] for int in interventions),
            max(int['end_time'] for int in interventions) + 1
        )

        effects = {year: self.simulator._get_intervention_effects() for year in years}

        for effect_type in ['emission_reduction', 'renewable_energy', 'carbon_capture']:
            effect_values = [effects[year][effect_type] for year in years]
            ax_effects.plot(years, effect_values, label=effect_type.replace('_', ' ').title(),
                          color=colors.get(effect_type, 'gray'))

        ax_effects.set_xlabel('Year')
        ax_effects.set_ylabel('Effect Intensity')
        ax_effects.set_title('Cumulative Intervention Effects')
        ax_effects.legend()
        ax_effects.grid(True)

        plt.tight_layout()

        if not show_timeline:
            return fig_timeline

        # Create figure for climate impact
        fig_impact = plt.figure(figsize=(12, 8))
        gs_impact = plt.GridSpec(3, 1)

        # Temperature impact
        ax_temp = fig_impact.add_subplot(gs_impact[0])
        temp_data = self.simulator.temperature.mean(dim=(0, 1)).cpu().numpy()
        ax_temp.plot(years, temp_data, label='Temperature', color='red')
        ax_temp.set_ylabel('Temperature (°C)')
        ax_temp.legend()
        ax_temp.grid(True)

        # Greenhouse gas impact
        ax_ghg = fig_impact.add_subplot(gs_impact[1])
        ghg_data = self.simulator.greenhouse_gas.mean(dim=(0, 1)).cpu().numpy()
        ax_ghg.plot(years, ghg_data, label='Greenhouse Gas', color='brown')
        ax_ghg.set_ylabel('Concentration (ppm)')
        ax_ghg.legend()
        ax_ghg.grid(True)

        # Combined effects
        ax_combined = fig_impact.add_subplot(gs_impact[2])
        for effect_type, color in colors.items():
            effect_values = [effects[year][effect_type] for year in years]
            ax_combined.plot(years, effect_values, label=effect_type.replace('_', ' ').title(),
                           color=color, alpha=0.6)
        ax_combined.set_xlabel('Year')
        ax_combined.set_ylabel('Effect Intensity')
        ax_combined.legend()
        ax_combined.grid(True)

        plt.tight_layout()

        return fig_timeline, fig_impact

    def plot_spatial_intervention_effects(self, year: Optional[int] = None) -> plt.Figure:
        """Plot the spatial distribution of intervention effects."""
        if year is None:
            year = self.simulator.year

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)

        # Temperature change plot
        ax_temp = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        temp_data = self.simulator.temperature.cpu().numpy()
        temp_change = temp_data - self.simulator.config.planet.initial_surface_temp_celsius

        im_temp = ax_temp.imshow(
            temp_change,
            transform=ccrs.PlateCarree(),
            origin='lower',
            extent=[-180, 180, -90, 90],
            cmap='RdBu_r'
        )
        ax_temp.add_feature(cfeature.COASTLINE)
        ax_temp.add_feature(cfeature.BORDERS, linestyle=':')
        plt.colorbar(im_temp, ax=ax_temp, label='Temperature Change (°C)')
        ax_temp.set_title('Temperature Impact')

        # Greenhouse gas change plot
        ax_ghg = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ghg_data = self.simulator.greenhouse_gas.cpu().numpy()
        ghg_change = ghg_data - self.simulator.config.planet.initial_greenhouse_gas_ppm

        im_ghg = ax_ghg.imshow(
            ghg_change,
            transform=ccrs.PlateCarree(),
            origin='lower',
            extent=[-180, 180, -90, 90],
            cmap='YlOrBr'
        )
        ax_ghg.add_feature(cfeature.COASTLINE)
        ax_ghg.add_feature(cfeature.BORDERS, linestyle=':')
        plt.colorbar(im_ghg, ax=ax_ghg, label='GHG Change (ppm)')
        ax_ghg.set_title('Greenhouse Gas Impact')

        # Intervention effects summary
        ax_summary = fig.add_subplot(gs[1, :])
        effects = self.simulator._get_intervention_effects()

        effect_names = list(effects.keys())
        effect_values = list(effects.values())

        colors = ['green', 'blue', 'purple']
        bars = ax_summary.bar(effect_names, effect_values, color=colors)

        ax_summary.set_ylabel('Effect Intensity')
        ax_summary.set_title(f'Intervention Effects Summary (Year {year})')
        ax_summary.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_summary.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}',
                          ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def plot_historical_simulation(self, years, temperatures, co2_levels, ph_levels,
                                historical_temps, historical_co2, historical_ph,
                                projected_temps, projected_co2, projected_ph,
                                output_dir: Optional[str] = None) -> str:
        """
        Plot historical simulation results with validation data.

        Args:
            years: List of years in the simulation
            temperatures: List of simulated temperatures
            co2_levels: List of simulated CO2 levels
            ph_levels: List of simulated pH levels
            historical_temps: Dict of historical temperature data
            historical_co2: Dict of historical CO2 data
            historical_ph: Dict of historical pH data
            projected_temps: Dict of projected temperature data
            projected_co2: Dict of projected CO2 data
            projected_ph: Dict of projected pH data
            output_dir: Optional output directory. If None, creates a timestamped directory.

        Returns:
            str: Path to the output directory containing all generated files
        """
        # Create output directory with timestamp if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("outputs", f"historical_simulation_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Create main figure with all variables
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Temperature plot
        ax1.plot(years, temperatures, 'b-', label='Simulated')
        ax1.scatter(list(historical_temps.keys()), list(historical_temps.values()),
                   color='red', label='Historical', alpha=0.6)
        ax1.scatter(list(projected_temps.keys()), list(projected_temps.values()),
                   color='green', label='Projected', alpha=0.6)
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Global Temperature Change')
        ax1.grid(True)
        ax1.legend()

        # CO2 plot
        ax2.plot(years, co2_levels, 'r-', label='Simulated')
        ax2.scatter(list(historical_co2.keys()), list(historical_co2.values()),
                   color='red', label='Historical', alpha=0.6)
        ax2.scatter(list(projected_co2.keys()), list(projected_co2.values()),
                   color='green', label='Projected', alpha=0.6)
        ax2.set_ylabel('CO2 (ppm)')
        ax2.set_title('Atmospheric CO2 Concentration')
        ax2.grid(True)
        ax2.legend()

        # pH plot
        ax3.plot(years, ph_levels, 'g-', label='Simulated')
        ax3.scatter(list(historical_ph.keys()), list(historical_ph.values()),
                   color='red', label='Historical', alpha=0.6)
        ax3.scatter(list(projected_ph.keys()), list(projected_ph.values()),
                   color='green', label='Projected', alpha=0.6)
        ax3.set_ylabel('Ocean pH')
        ax3.set_title('Ocean pH Change')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()

        # Save the main plot
        main_plot_path = os.path.join(output_dir, 'historical_simulation.png')
        plt.savefig(main_plot_path)
        plt.close()

        # Save individual plots
        for i, (data, title) in enumerate([(temperatures, 'temperature'),
                                         (co2_levels, 'co2'),
                                         (ph_levels, 'ph')]):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(years, data, 'b-', label='Simulated')
            ax.scatter(list(historical_temps.keys()), list(historical_temps.values()),
                      color='red', label='Historical', alpha=0.6)
            ax.scatter(list(projected_temps.keys()), list(projected_temps.values()),
                      color='green', label='Projected', alpha=0.6)
            ax.set_ylabel(title.capitalize())
            ax.set_title(f'Global {title.capitalize()} Change')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{title}_change.png'))
            plt.close()

        # Save simulation data
        data = {
            'years': years,
            'temperatures': temperatures,
            'co2_levels': co2_levels,
            'ph_levels': ph_levels,
            'historical_temps': historical_temps,
            'historical_co2': historical_co2,
            'historical_ph': historical_ph,
            'projected_temps': projected_temps,
            'projected_co2': projected_co2,
            'projected_ph': projected_ph
        }
        with open(os.path.join(output_dir, 'simulation_data.json'), 'w') as f:
            json.dump(data, f, indent=2)

        return output_dir