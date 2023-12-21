import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Lie",
    version="0.0.1",
    author="Thomas Firmin",
    author_email="thomas.firmin@univ-lille.fr",
    description="An addon framework for BindsNET",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CeCILL",
        "Operating System :: OS Independent",
    ],
    keywords=["SNN", "spiking", "optimization"],
    package_dir={"": "lib"},
    packages=setuptools.find_packages("lib"),
    install_requires=[
        "numpy",
        "bindsnet @ git+https://github.com/BindsNET/bindsnet.git@master#egg=bindsnet",
        "matplotlib",
        "torch",
        "pandas",
    ],
    extras_require={},
    python_requires=">=3.8",
)
