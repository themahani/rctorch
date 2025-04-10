from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-force",
    version="0.1.0",
    author="Ali Mahani",
    author_email="ali.a.mahani@zoho.com",
    description="Morris-Lecar neural networks with force learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/themahani/ml-force",
    project_urls={
        "Bug Tracker": "https://github.com/themahani/ml-force/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.8.0",
        "matplotlib>=3.3.0",
        "scipy>=1.6.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
    },
)
