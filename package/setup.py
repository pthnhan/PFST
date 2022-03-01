import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pfs",
    version="0.0.1",
    author="pthnhan",
    author_email="nhanmath97@gmail.com",
    description="Parallel feature selection based on Trace ratio criterion!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["pfs"] + setuptools.find_packages(include=["pfs.*", "pfs"]),
    package_dir={"pfs": "pfs"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
