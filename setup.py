from setuptools import setup, find_packages

setup(
    name="wildfire_modeling",
    version="0.1.0",
    description="Wildfire modeling utilities and analysis tools",
    author="Luke von Kapff",
    packages=find_packages(include=["utils", "utils.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "geopandas",
        "powerlaw"
    ],
    python_requires=">=3.9",
)
