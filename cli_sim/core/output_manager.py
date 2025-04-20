import os
from datetime import datetime
from pathlib import Path
from typing import Optional

class OutputManager:
    def __init__(self, simulation_name: Optional[str] = None, base_output_dir: Optional[str] = None):
        """Initialize the output manager for a simulation run.

        Args:
            simulation_name: Optional name for this simulation run. If None, uses timestamp.
            base_output_dir: Optional base directory for outputs. If None, uses 'outputs'.
        """
        self.simulation_name = simulation_name or "simulation"
        self.base_output_dir = Path(base_output_dir or "outputs")
        self.simulation_dir = self._create_simulation_directory()

    def _create_simulation_directory(self) -> Path:
        """Create a unique directory for this simulation run."""
        # Create base outputs directory if it doesn't exist
        self.base_output_dir.mkdir(exist_ok=True, parents=True)

        # Create timestamped directory for this simulation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulation_dir = self.base_output_dir / f"{self.simulation_name}_{timestamp}"
        simulation_dir.mkdir(exist_ok=True, parents=True)

        return simulation_dir

    def get_output_path(self, filename: str) -> Path:
        """Get the full path for a file in the simulation directory.

        Args:
            filename: Name of the output file

        Returns:
            Path object for the output file
        """
        return self.simulation_dir / filename

    def get_output_directory(self) -> Path:
        """Get the path to the simulation directory.

        Returns:
            Path object for the output directory
        """
        return self.simulation_dir

    def get_relative_path(self, filename: str) -> str:
        """Get the relative path for a file in the simulation directory.

        Args:
            filename: Name of the output file

        Returns:
            String containing the relative path to the output file
        """
        return str(self.simulation_dir / filename)

    def get_relative_directory(self) -> str:
        """Get the relative path to the simulation directory.

        Returns:
            String containing the relative path to the output directory
        """
        return str(self.simulation_dir)