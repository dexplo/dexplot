import setuptools

with open('dexplot/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split("'")[1]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dexplot",
    version=version,
    author="Ted Petrou",
    author_email="petrou.theodore@gmail.com",
    description="Powerful and intuitive data visualization library using matplotlib for both long and wide data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="data visualization matplotlib pandas",
    url="https://github.com/dexplo/dexplot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Framework :: Matplotlib"
    ],
    install_requires=['numpy>=1.15',
                      'scipy>=1.0'
                      'matplotlib>=3.1', 
                      'pandas>=0.24'],
    extras_require={
        "apps":  ["ipywidgets"],
    },
    python_requires='>=3.6'
)