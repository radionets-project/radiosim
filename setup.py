from setuptools import setup, find_packages

setup(
    name="radiosim",
    version="0.0.1",
    description="Simulation of radio skies to create astrophysical data sets",
    url="https://github.com/Kevin2/radionets",
    author="Kevin Schmidt, Felix Geyer, Paul-Simon Blomenkamp, Stefan Fröse",
    author_email="kevin3.schmidt@tu-dortmund.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "h5py"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    zip_safe=False,
    # entry_points={
    #     "console_scripts": [
    #         "radiosim = radiosim.scripts.start_simulation:main",
    #     ],
    # },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)