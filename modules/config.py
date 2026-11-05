"""Project paths and run settings for the Energy Policy model."""

from __future__ import annotations

import os.path as op
import importlib.metadata as importlib_metadata
import re
import subprocess
import sys
from pathlib import Path

# ============================================================================
# PATH DETECTION
# ============================================================================



# Detect the repository root from this file, independently of the current
# working directory.  ``config.py`` lives directly below the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = str(PROJECT_ROOT / "modules")
CALC_ROOT = PROJECT_ROOT

# ============================================================================
# SOURCE DATA PATHS (reorganized structure)
# ============================================================================

# Main directory
MODULES_ROOT = CALC_ROOT / "modules"

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
MRIO_DATA_DIR = DATASETS_DIR / "mrio"
EMPLOYMENT_DATA_DIR = DATASETS_DIR / "employment"
HELPERS_DIR = MRIO_ROOT / "helpers"
METADATA_DIR = MRIO_ROOT / "metadata"
MRIO_SCENARIOS_DIR = MRIO_ROOT / "scenarios"

# --- Generated results ---
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_IMAGES = RESULTS_ROOT / "images"
RESULTS_TABLES = RESULTS_ROOT / "tables"
RESULTS_SUPPLEMENTARY = RESULTS_ROOT / "supplementary"
RESULTS_CACHE = RESULTS_SUPPLEMENTARY / "cache"

# Keep the established names used by plotting and table helpers, but point
# them at the repository-local output tree.
IMG_ROOT = RESULTS_IMAGES
TAB_ROOT = RESULTS_TABLES

for _output_dir in (RESULTS_IMAGES, RESULTS_TABLES, RESULTS_SUPPLEMENTARY):
    _output_dir.mkdir(parents=True, exist_ok=True)


def project_relative_path(path: str | Path) -> str:
    """Return a portable project-relative display path for logs and notebooks."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()

# ============================================================================
# FIGURE SETTINGS
# ============================================================================

# Default values for figure saving/closing behavior
# Can be overridden in notebooks
CLOSE_FIGURES = True
SAVE_FIGURES = False
INSTALL_MISSING_REQUIREMENTS = False

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


def _status_marker(preferred: str, fallback: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        preferred.encode(encoding)
        return preferred
    except UnicodeEncodeError:
        return fallback


def verify_requirements(
    requirements_path: Path | str | None = None,
    *,
    install_missing: bool | None = None,
) -> None:
    """Check that packages listed in requirements.txt are available."""
    if requirements_path is None:
        requirements_path = CALC_ROOT / "requirements.txt"
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
    ok_marker = _status_marker("✓", "OK")
    print("Checking Python requirements:")
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
