import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pfst",
    version="0.1.0",
    author="pthnhan",
    author_email="nhanmath97@gmail.com",
    description="Parallel feature selection based on Trace ratio criterion!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["pfst"] + setuptools.find_packages(include=["pfst.*", "pfst"]),
    package_dir={"pfst": "pfst"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
