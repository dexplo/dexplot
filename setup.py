import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

pat = r'!\[png\]\('
repl = r'![png](https://raw.githubusercontent.com/dexplo/dexplot/master/'
long_description = re.sub(pat, repl, long_description)

setuptools.setup(
    name="dexplot",
    version="0.0.6",
    author="Ted Petrou",
    author_email="petrou.theodore@gmail.com",
    description="Simple plotting library for both long and wide data integrated with DataFrames",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dexplo/dexplot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)