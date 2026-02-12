from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent


def discover_packages() -> list[str]:
    namespace_packages = find_namespace_packages(where=".")
    return ["evenet", *[f"evenet.{name}" for name in namespace_packages]]


setup(
    name="evenet-core",
    version="0.1.0",
    description="Core components for EveNet models.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    author="EveNet contributors",
    license="MIT",
    packages=discover_packages(),
    package_dir={"evenet": "."},
    include_package_data=True,
    install_requires=[
        "lightning>=2.0",
        "matplotlib>=3.7",
        "numpy>=1.24",
        "opt-einsum>=3.3",
        "pyarrow>=14.0",
        "pyyaml>=6.0",
        "rich>=13.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "sympy>=1.12",
        "torch>=2.1",
        # "torch-linear-assignment>=0.0.5",
        "tqdm>=4.65",
        "wandb>=0.16",
    ],
    keywords=["machine-learning", "physics", "pytorch", "evenet"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
