"""Setup script for lattice-gaussian-mcmc package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lattice-gaussian-mcmc",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Lattice Gaussian MCMC sampling based on Wang & Ling (2018)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lattice-gaussian-mcmc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "memory-profiler>=0.58.0",
            "line-profiler>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "nbformat>=5.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lattice-mcmc=experiments.scripts.run_all_experiments:main",
        ],
    },
)