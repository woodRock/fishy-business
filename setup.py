from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="fishy-business",
    version="0.1.0",
    author="Jesse Wood",
    author_email="jesse.wood@ecs.vuw.ac.nz",
    description="A configuration-driven framework for spectral data analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/woodRock/fishy-business",
    packages=find_packages(exclude=["tests*", "dashboard*"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "viz": ["pygraphviz"],
    },
    entry_points={
        "console_scripts": [
            "fishy=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
