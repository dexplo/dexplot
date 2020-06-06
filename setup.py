import setuptools

from dexplot import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dexplot",
    version=__version__,
    author="Ted Petrou",
    author_email="petrou.theodore@gmail.com",
    description="Data Visualization library using matplotlib for both long and wide data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="visualization matplotlib pandas",
    url="https://github.com/dexplo/dexplot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=['matplotlib', 'pandas'],
    python_requires='>=3.6'
)