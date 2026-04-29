import init
import setuptools

with open("README.md", "r", encoding="UTF-8") as file:
    long_description = file.read()

requirements = []
with open("requirements.lib.txt", "r", encoding="UTF-8") as file:
    for line in file:
        requirements.append(line.strip())


version = init.read_version()
init.write_version(version)

setuptools.setup(
    name="stem_pytorch",
    version=version,
    author="STEM-DFER Team",
    author_email="stem_dfer@example.com",
    description="Official pytorch implementation for STEM-DFER.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/STEM-DFER/STEM-DFER",
    keywords=["deep learning", "pytorch", "AI", "facial expression recognition", "DFER"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["stem_pytorch", "stem_pytorch.*"]),
    package_data={
        "stem_pytorch": [
            "version.txt"
        ]
    },
    python_requires=">=3.6",
    install_requires=requirements,
    license="CC BY-NC 4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
)
