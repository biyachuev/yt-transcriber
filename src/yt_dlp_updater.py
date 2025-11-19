"""
yt-dlp version checker and updater module.

This module checks if yt-dlp is up to date and automatically updates it if needed.
YouTube frequently changes their API, so keeping yt-dlp updated is crucial.
"""

import subprocess
import sys
from packaging import version
from src.logger import logger


def get_current_version() -> str:
    """
    Get the currently installed version of yt-dlp.

    Returns:
        str: Current version string (e.g., "2025.10.22")
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "yt-dlp"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
        return None
    except subprocess.CalledProcessError:
        return None


def get_latest_version() -> str:
    """
    Get the latest available version of yt-dlp from PyPI.

    Returns:
        str: Latest version string, or None if check fails
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "yt-dlp"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        # Parse output to find latest version
        # Format: "yt-dlp (2025.10.22)"
        for line in result.stdout.split("\n"):
            if "Available versions:" in line or "LATEST:" in line:
                # Extract version from line
                parts = line.split()
                for part in parts:
                    if part[0].isdigit():
                        return part.rstrip(",")

        # Alternative approach: check first line which usually contains latest
        first_line = result.stdout.split("\n")[0]
        if "(" in first_line and ")" in first_line:
            ver = first_line.split("(")[1].split(")")[0].strip()
            if ver[0].isdigit():
                return ver

        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return None


def update_yt_dlp() -> bool:
    """
    Update yt-dlp to the latest version.

    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        logger.info("Updating yt-dlp to the latest version...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        logger.info("yt-dlp updated successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Failed to update yt-dlp: {e}")
        return False


def check_and_update_yt_dlp(force_update: bool = False) -> bool:
    """
    Check if yt-dlp needs updating and update if necessary.

    Args:
        force_update: If True, update regardless of version check

    Returns:
        bool: True if yt-dlp is up to date (or was successfully updated), False otherwise
    """
    current = get_current_version()

    if not current:
        logger.warning("Could not determine current yt-dlp version")
        return False

    logger.info(f"Current yt-dlp version: {current}")

    if force_update:
        logger.info("Force update requested")
        return update_yt_dlp()

    # Get latest version
    latest = get_latest_version()

    if not latest:
        logger.warning(
            "Could not check for latest yt-dlp version, skipping update check"
        )
        logger.info("Continuing with current version...")
        return True  # Don't block execution if we can't check

    logger.info(f"Latest yt-dlp version: {latest}")

    # Compare versions
    try:
        if version.parse(current) < version.parse(latest):
            logger.info(f"yt-dlp update available: {current} -> {latest}")
            return update_yt_dlp()
        else:
            logger.info("yt-dlp is up to date")
            return True
    except Exception as e:
        logger.warning(f"Could not compare versions: {e}")
        logger.info("Continuing with current version...")
        return True  # Don't block execution if version comparison fails
