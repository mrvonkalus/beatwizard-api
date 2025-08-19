#!/usr/bin/env python3
"""
Setup script for BeatWizard package
"""

from setuptools import setup, find_packages

setup(
    name="beatwizard",
    version="1.0.0",
    description="Advanced AI-powered music production analysis and feedback",
    author="BeatWizard Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "numpy>=1.20.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "pyloudnorm>=0.1.0",
        "scipy>=1.10.0",
        "openai>=1.0.0",
        "loguru>=0.7.0",
        "supabase>=2.0.0"
    ],
    extras_require={
        "full": [
            "pydub>=0.25.1",
            "audioread>=3.0.0",
            "numba>=0.58.0",
            "llvmlite>=0.41.0"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
