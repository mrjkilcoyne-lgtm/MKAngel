"""MKAngel — Grammar Language Model"""

from setuptools import setup, find_packages

setup(
    name="mkangel",
    version="0.1.0",
    description=(
        "A Grammar Language Model that learns the deep structural rules "
        "underlying natural languages, chemistry, biology, and code. "
        "Learns scales to play masterpieces."
    ),
    author="mrjkilcoyne",
    author_email="mrjkilcoyne@gmail.com",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "mkangel=glm.angel:main",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
