import os
from datetime import datetime
from pathlib import Path

class OutputManager:
    def __init__(self, simulation_name: str):
        """Initialize the output manager for a simulation run."""
        self.simulation_name = simulation_name
        self.base_output_dir = Path("outputs")
        self.simulation_dir = self._create_simulation_directory()

    def _create_simulation_directory(self) -> Path:
        """Create a unique directory for this simulation run."""
        # Create base outputs directory if it doesn't exist
        self.base_output_dir.mkdir(exist_ok=True)

        # Create timestamped directory for this simulation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        simulation_dir = self.base_output_dir / f"{self.simulation_name}_{timestamp}"
        simulation_dir.mkdir(exist_ok=True)

        return simulation_dir

    def get_path(self, filename: str) -> str:
        """Get the full path for a file in the simulation directory."""
        return str(self.simulation_dir / filename)

    def get_directory(self) -> str:
        """Get the path to the simulation directory."""
        return str(self.simulation_dir)