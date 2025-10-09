
from setuptools import setup, find_packages

setup(
    name="wildfire_powerlaw",
    version="0.2.0",
    description="Modified powerlaw with Weibull, GPD, bootstrap SEs, and streamlined outputs",
    author="Your Name",
    packages=find_packages(),
    install_requires=["numpy","scipy","pandas"],
    python_requires=">=3.8",
)
