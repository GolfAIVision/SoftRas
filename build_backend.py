"""
Custom build backend that ensures correct CUDA version is used.
This wraps setuptools.build_meta and sets CUDA_HOME before building.
"""

import os
import sys
from pathlib import Path
from tabnanny import verbose
from setuptools import build_meta as _orig


def get_major_minor_from_string(version_str: str) -> tuple[int, ...]:
    parts = tuple([int(p) for p in version_str.split(".")])
    assert len(parts) >= 1, f"Must have at least major version, got {version_str}."
    return parts


def _get_pytorch_cuda_version() -> tuple[int, ...] | None:
    """Get the CUDA version that PyTorch was compiled with

    Returns:
        major, minor iff PyTorch found, None otherwise.
    """
    try:
        import torch

        if cuda_version := torch.version.cuda:
            return get_major_minor_from_string(cuda_version)
    except (ImportError, AttributeError, ValueError):
        return None


def _find_cuda():
    """
    Find CUDA installation compatible with PyTorch.
    Detects PyTorch's CUDA version and finds matching CUDA toolkit.

    We assume the happy path, since we control the computers this code will run on.
    So: there must be a major.minor version that matches the PyTorch version.

    """
    usr_local = Path("/usr/local")
    if not usr_local.exists():
        return None

    # Get PyTorch's CUDA version requirement
    cuda_dirs: list[tuple[tuple[int, ...], str]] = []

    # Scan all CUDA installations
    for path in usr_local.glob("cuda-*"):
        if path.is_dir() and (path / "bin" / "nvcc").exists():
            version_str = path.name.replace("cuda-", "")
            try:
                version = get_major_minor_from_string(version_str)
                cuda_dirs.append((version, str(path)))
            except ValueError:
                pass

    if pytorch_cuda := _get_pytorch_cuda_version():
        print(
            f"build_backend: PyTorch compiled with CUDA {pytorch_cuda}", file=sys.stderr
        )
        # Try to find exact match for PyTorch's CUDA version
        for version, path in cuda_dirs:
            if version == pytorch_cuda:
                print(
                    f"build_backend: Found exact CUDA match {version}", file=sys.stderr
                )
                return path

        # Try to find same major.minor version
        for version, path in cuda_dirs:
            major, minor, *_ = version
            pytorch_major, pytorch_minor, *_ = pytorch_cuda

            if major == pytorch_major and minor == pytorch_minor:
                print(
                    f"build_backend: Found compatible CUDA {version} for PyTorch {pytorch_cuda}",
                    file=sys.stderr,
                )
                return path

    # Give up if no match
    print(f"Can't find exact match for CUDA PyTorch was built with - build may fail.")
    return None


def _setup_cuda_env():
    """Setup CUDA environment variables"""
    if "CUDA_HOME" not in os.environ:
        cuda_home = _find_cuda()
        if cuda_home:
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["CUDA_PATH"] = cuda_home
            # Prepend CUDA bin to PATH
            cuda_bin = os.path.join(cuda_home, "bin")
            path_parts = [p for p in os.environ.get("PATH", "").split(":") if p]
            # Remove old CUDA paths and add the correct one first
            path_parts = [p for p in path_parts if "/cuda" not in p or cuda_bin in p]
            path_parts.insert(0, cuda_bin)
            os.environ["PATH"] = ":".join(path_parts)
            print(f"build_backend: Using CUDA from {cuda_home}", file=sys.stderr)


# Wrap all build functions to ensure CUDA is setup
def get_requires_for_build_wheel(config_settings=None):
    _setup_cuda_env()
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _setup_cuda_env()
    return _orig.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _setup_cuda_env()
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _setup_cuda_env()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _setup_cuda_env()
    return _orig.build_sdist(sdist_directory, config_settings)


# For editable installs
if hasattr(_orig, "get_requires_for_build_editable"):

    def get_requires_for_build_editable(config_settings=None):
        _setup_cuda_env()
        return _orig.get_requires_for_build_editable(config_settings)

    def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
        _setup_cuda_env()
        return _orig.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )

    def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
        _setup_cuda_env()
        # Debug: print current PATH
        print(
            f"build_backend.build_editable: PATH={os.environ.get('PATH', '')[:200]}...",
            file=sys.stderr,
        )
        return _orig.build_editable(
            wheel_directory, config_settings, metadata_directory
        )
