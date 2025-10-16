"""
Setup script for installing the package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description.
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Load runtime requirements.
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="yt-transcriber",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Universal toolkit for transcribing and translating videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yt-transcriber",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "yt-transcriber=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
