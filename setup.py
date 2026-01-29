from setuptools import setup, find_packages

setup(
    name="kineticstack",
    version="0.1.0",
    description="Self-evolving DL stack using Agentic Software Engineering",
    author="KineticStack Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
        ],
    },
)
