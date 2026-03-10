"""MKAngel — Grammar Language Model"""

from setuptools import setup, find_packages

setup(
    name="mkangel",
    version="0.2.0",
    description=(
        "A Grammar Language Model that learns the deep structural rules "
        "underlying natural languages, chemistry, biology, code, mathematics, "
        "and physics.  Speaks, listens, swarms, and self-improves.  "
        "Learns scales to play masterpieces."
    ),
    author="mrjkilcoyne",
    author_email="mrjkilcoyne@gmail.com",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
        "voice": [
            "sounddevice>=0.4",
            "numpy>=1.24",
            "openai-whisper>=20230918",
            "pyttsx3>=2.90",
            "scipy>=1.11",
        ],
        "voice-neural": [
            "TTS>=0.20",
        ],
        "cloud": [
            "boto3>=1.28",
            "requests>=2.31",
        ],
        "all": [
            "sounddevice>=0.4",
            "numpy>=1.24",
            "openai-whisper>=20230918",
            "pyttsx3>=2.90",
            "scipy>=1.11",
            "boto3>=1.28",
            "requests>=2.31",
        ],
    },
    entry_points={
        "console_scripts": [
            "mkangel=app.main:main",
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
