#!/usr/bin/env python3
"""
Setup script for HEALRAG library
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="healrag",
    version="0.1.0",
    author="HEALRAG Team",
    author_email="team@healrag.com",
    description="Azure RAG Library with Blob Storage and MarkItDown integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/healrag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Markup",
        "Topic :: Database",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "healrag=healraglib.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="azure, rag, blob-storage, document-processing, markitdown, ai, ml",
    project_urls={
        "Bug Reports": "https://github.com/your-org/healrag/issues",
        "Source": "https://github.com/your-org/healrag",
        "Documentation": "https://github.com/your-org/healrag#readme",
    },
) 