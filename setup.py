import os
import sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_and_set_cuda():
    """
    Find and set CUDA_HOME if not already set.
    Prefers /usr/local/cuda (system default) or CUDA >= 12.8.

    Note: This runs at import time, but the build system may use an isolated
    environment. When installing as a dependency, users should set CUDA_HOME
    in their project's pyproject.toml using [tool.uv.extra-build-variables].
    """
    if "CUDA_HOME" in os.environ:
        cuda_home = os.environ["CUDA_HOME"]
        if Path(cuda_home, "bin", "nvcc").exists():
            print(f"Using CUDA_HOME from environment: {cuda_home}", file=sys.stderr)
            return

    # Check /usr/local/cuda first (system default)
    cuda_default = Path("/usr/local/cuda")
    if cuda_default.exists() and (cuda_default / "bin" / "nvcc").exists():
        cuda_home = str(cuda_default.resolve())
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
        print(f"Using system default CUDA: {cuda_home}", file=sys.stderr)
        return

    # Fall back to searching for CUDA >= 12.8
    usr_local = Path("/usr/local")
    if usr_local.exists():
        cuda_dirs: list[tuple[float, str]] = []
        for path in usr_local.glob("cuda-*"):
            if path.is_dir() and (path / "bin" / "nvcc").exists():
                version_str = path.name.replace("cuda-", "")
                try:
                    version = float(version_str)
                    if version >= 12.8:
                        cuda_dirs.append((version, str(path)))
                except ValueError:
                    pass

        if cuda_dirs:
            # Use the highest version >= 12.8
            cuda_dirs.sort(reverse=True)
            cuda_home = cuda_dirs[0][1]
            os.environ["CUDA_HOME"] = cuda_home
            os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
            print(f"Found CUDA {cuda_dirs[0][0]}: {cuda_home}", file=sys.stderr)
            return

    print("WARNING: Could not find CUDA >= 12.8. Build may fail.", file=sys.stderr)
    print(
        "Set CUDA_HOME environment variable to specify CUDA location.", file=sys.stderr
    )


# Try to find and set CUDA before building
find_and_set_cuda()


# CUDA extension modules
ext_modules = [
    CUDAExtension(
        "soft_renderer.cuda.load_textures",
        [
            "soft_renderer/cuda/load_textures_cuda.cpp",
            "soft_renderer/cuda/load_textures_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        "soft_renderer.cuda.create_texture_image",
        [
            "soft_renderer/cuda/create_texture_image_cuda.cpp",
            "soft_renderer/cuda/create_texture_image_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        "soft_renderer.cuda.soft_rasterize",
        [
            "soft_renderer/cuda/soft_rasterize_cuda.cpp",
            "soft_renderer/cuda/soft_rasterize_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        "soft_renderer.cuda.voxelization",
        [
            "soft_renderer/cuda/voxelization_cuda.cpp",
            "soft_renderer/cuda/voxelization_cuda_kernel.cu",
        ],
    ),
]

# Setup configuration
setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExtension})
