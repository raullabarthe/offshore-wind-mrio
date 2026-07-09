"""Project paths and run settings for the Energy Policy model."""

from __future__ import annotations

import os
import os.path as op
import importlib.metadata as importlib_metadata
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ============================================================================
# PATH DETECTION
# ============================================================================

def _first_existing(*candidates: str) -> Optional[str]:
    """Return the first existing directory from candidates."""
    return next((c for c in candidates if op.isdir(c)), None)


# Detect module and project roots (safe in notebooks and scripts).
ROOT = Path(op.dirname(op.abspath(__file__))) if "__file__" in globals() else Path.cwd()
MODULES_ROOT = ROOT if ROOT.name == "modules" else ROOT / "modules"
PROJECT_ROOT = MODULES_ROOT.parent

# Generated files are written under outputs/ so source inputs and outputs stay separate.
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
IMG_ROOT = OUTPUTS_ROOT / "figures"
TAB_ROOT = OUTPUTS_ROOT / "tables"
CACHE_ROOT = OUTPUTS_ROOT / "cache"
RESULTS_ROOT = CACHE_ROOT / "results"
MRIO_SCENARIOS_DIR = CACHE_ROOT / "mrio_scenarios"

# ============================================================================
# SOURCE DATA PATHS (reorganized structure)
# ============================================================================

# Main directory

# --- Cost model module ---
COST_MODEL_ROOT = MODULES_ROOT / "cost_model"
COST_SOURCES = COST_MODEL_ROOT / "sources"
EXCHANGE_ROOT = COST_SOURCES / "exchange_rates"

# --- Supply chain module ---
SUPPLY_CHAIN_ROOT = MODULES_ROOT / "supply_chain"
SUPPLY_SOURCES = SUPPLY_CHAIN_ROOT / "sources"
TURBINE_DATA_ROOT = SUPPLY_SOURCES / "turbine"
WINDFARM_DATA_ROOT = SUPPLY_SOURCES / "wind_farm"

# --- MRIO analysis sources ---
MRIO_ROOT = MODULES_ROOT / "mrio_analysis"
DATASETS_DIR = MRIO_ROOT / "datasets"
HELPERS_DIR = MRIO_ROOT / "helpers"
METADATA_DIR = MRIO_ROOT / "metadata"
for _path in (OUTPUTS_ROOT, IMG_ROOT, TAB_ROOT, CACHE_ROOT, RESULTS_ROOT, MRIO_SCENARIOS_DIR):
    _path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FIGURE SETTINGS
# ============================================================================

# Default values for figure saving/closing behavior
# Can be overridden in notebooks
CLOSE_FIGURES = True
SAVE_FIGURES = False
INSTALL_MISSING_REQUIREMENTS = True
REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 13

# ============================================================================
# DEPLOYMENT COUNTRIES
# ============================================================================

# Core European deployment countries for analysis
CORE_DEPLOYMENT = ["GB", "BE", "NL", "DK", "DE"]


# ============================================================================
# ENVIRONMENT CHECKS
# ============================================================================

def _requirement_package_name(line: str) -> str | None:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "http://", "https://", "git+")):
        return None
    line = line.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)", line)
    return match.group(1) if match else None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:4])


def _requirement_is_satisfied(line: str, installed_version: str) -> bool:
    spec = line.split("#", 1)[0].split(";", 1)[0].strip()
    match = re.search(r"(>=|==|<=|>|<)\s*([A-Za-z0-9_.!+-]+)", spec)
    if not match:
        return True
    op, required_version = match.groups()
    current = _version_tuple(installed_version)
    required = _version_tuple(required_version)
    if op == ">=":
        return current >= required
    if op == "==":
        return current == required
    if op == "<=":
        return current <= required
    if op == ">":
        return current > required
    if op == "<":
        return current < required
    return True


def verify_python_version() -> None:
    """Require the Python version used to validate the public workflow."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    if (major, minor) != (REQUIRED_PYTHON_MAJOR, REQUIRED_PYTHON_MINOR):
        raise RuntimeError(
            "This model is configured for Python "
            f"{REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}.x. "
            f"The active environment is Python {major}.{minor}. "
            "Create/select a Python 3.13 environment before running the notebook."
        )


def verify_requirements(
    requirements_path: Path | str | None = None,
    *,
    install_missing: bool | None = None,
) -> None:
    """Check that packages listed in requirements.txt are available."""
    verify_python_version()

    if requirements_path is None:
        requirements_path = PROJECT_ROOT / "requirements.txt"
    requirements_path = Path(requirements_path)
    if install_missing is None:
        install_missing = INSTALL_MISSING_REQUIREMENTS

    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")

    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if _requirement_package_name(line)
    ]

    missing = []
    ok_marker = "OK"
    print(f"Checking Python requirements with Python {sys.version_info.major}.{sys.version_info.minor}:")
    for requirement in requirements:
        package_name = _requirement_package_name(requirement)
        try:
            installed_version = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            print(f"  x {requirement}: not installed")
            missing.append(requirement)
            continue

        if _requirement_is_satisfied(requirement, installed_version):
            print(f"  {ok_marker} {requirement}: {installed_version}")
        else:
            print(f"  x {requirement}: installed {installed_version}")
            missing.append(requirement)

    if missing and install_missing:
        print("Installing missing requirements from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        return verify_requirements(requirements_path, install_missing=False)

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "Missing or incompatible Python requirements: "
            f"{missing_text}. Install them with: {sys.executable} -m pip install -r {requirements_path}"
        )
