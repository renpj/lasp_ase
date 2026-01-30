import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    'ase',
]
setuptools.setup(
    name="lasp_ase", 
    version="0.0.3",
    author="",
    description="ASE calculator for LASP", 
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/renpj/lasp_ase", 
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)