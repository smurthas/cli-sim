"""
Comprehensive validation script for the climate simulation system.

This script runs:
1. Unit tests for core functionality
2. Historical simulation validation
3. Visualization output checks
"""

import unittest
import os
import sys
import subprocess
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import re

def run_unit_tests():
    """Run all unit tests in the tests directory."""
    print("\nRunning unit tests...")
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('cli_sim/tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def parse_metric_value(value_str: str) -> float:
    """Parse a metric value from a string, handling units and special characters."""
    # Remove any non-numeric characters except decimal point and minus sign
    cleaned = re.sub(r'[^\d.-]', '', value_str)
    return float(cleaned)

def run_historical_simulation():
    """Run the historical simulation and validate its results."""
    print("\nRunning historical simulation...")
    try:
        # Run the historical simulation
        result = subprocess.run(
            [sys.executable, '-m', 'cli_sim.examples.historical_simulation'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Error running historical simulation:")
            print(result.stderr)
            return False

        # Parse the output to get RMSE values
        output = result.stdout
        historical_metrics = {}
        projected_metrics = {}

        for line in output.split('\n'):
            if 'Historical Period' in line:
                current_metrics = historical_metrics
            elif 'Projected Period' in line:
                current_metrics = projected_metrics
            elif 'RMSE' in line:
                metric, value = line.split(':')
                current_metrics[metric.strip()] = parse_metric_value(value)

        # Validate metrics against acceptable thresholds
        thresholds = {
            'historical': {
                'Temperature RMSE': 1.0,  # °C
                'CO2 RMSE': 100.0,  # ppm
                'Ocean pH RMSE': 0.1  # pH units
            },
            'projected': {
                'Temperature RMSE': 3.0,  # °C
                'CO2 RMSE': 250.0,  # ppm
                'Ocean pH RMSE': 0.3  # pH units
            }
        }

        # Check if all metrics are within thresholds
        validation_passed = True
        for period, metrics in [('historical', historical_metrics), ('projected', projected_metrics)]:
            print(f"\n{period.capitalize()} Period Validation:")
            for metric, value in metrics.items():
                threshold = thresholds[period][metric]
                status = "PASS" if value <= threshold else "FAIL"
                print(f"{metric}: {value:.2f} (threshold: {threshold:.2f}) - {status}")
                if value > threshold:
                    validation_passed = False

        return validation_passed

    except Exception as e:
        print(f"Error during historical simulation validation: {str(e)}")
        return False

def check_visualization_outputs():
    """Check that visualization outputs are being generated correctly."""
    print("\nChecking visualization outputs...")

    # Find the most recent output directory
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("No outputs directory found")
        return False

    # Get all historical simulation output directories
    sim_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("historical_simulation_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not sim_dirs:
        print("No historical simulation outputs found")
        return False

    latest_dir = sim_dirs[0]
    print(f"Checking outputs in: {latest_dir}")

    # Check for required files
    required_files = [
        'historical_simulation.png',
        'temperature_change.png',
        'co2_change.png',
        'ph_change.png',
        'simulation_data.json'
    ]

    all_files_exist = True
    for file in required_files:
        file_path = latest_dir / file
        if not file_path.exists():
            print(f"Missing required file: {file}")
            all_files_exist = False

    if all_files_exist:
        print("All required visualization files present")

        # Validate JSON data
        with open(latest_dir / 'simulation_data.json', 'r') as f:
            data = json.load(f)
            required_keys = [
                'years', 'temperatures', 'co2_levels', 'ph_levels',
                'historical_temps', 'historical_co2', 'historical_ph',
                'projected_temps', 'projected_co2', 'projected_ph'
            ]
            if all(key in data for key in required_keys):
                print("Simulation data JSON is valid")
            else:
                print("Simulation data JSON is missing required keys")
                all_files_exist = False

    return all_files_exist

def main():
    """Run all validation checks."""
    print("Starting comprehensive validation...")

    # Run unit tests
    tests_passed = run_unit_tests()

    # Run historical simulation validation
    simulation_passed = run_historical_simulation()

    # Check visualization outputs
    visualization_passed = check_visualization_outputs()

    # Print summary
    print("\nValidation Summary:")
    print(f"Unit Tests: {'PASSED' if tests_passed else 'FAILED'}")
    print(f"Historical Simulation: {'PASSED' if simulation_passed else 'FAILED'}")
    print(f"Visualization Outputs: {'PASSED' if visualization_passed else 'FAILED'}")

    # Overall status
    all_passed = tests_passed and simulation_passed and visualization_passed
    print(f"\nOverall Status: {'PASSED' if all_passed else 'FAILED'}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())