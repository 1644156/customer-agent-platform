# -*- coding: utf-8 -*-
"""Packaging configuration for the customer agent platform."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements(filename: str = "requirements.txt") -> list[str]:
    requirements_path = ROOT / filename
    if not requirements_path.exists():
        return []

    requirements: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


def read_version() -> str:
    init_path = ROOT / "customer_agent" / "__init__.py"
    if not init_path.exists():
        return "0.1.0"

    for line in init_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.1.0"


def read_long_description() -> str:
    readme_path = ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="customer_agent",
    version=read_version(),
    author="Li Qijun",
    description="Task-oriented customer service Agent platform powered by LLM workflows",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(
        include=["customer_agent", "customer_agent.*"],
        exclude=[
            "tests",
            "tests.*",
            "docs",
            "docs.*",
            "commerce_service_app",
            "commerce_service_app.*",
        ],
    ),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "customer-agent=customer_agent.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="customer-service agent llm langgraph rag fastapi",
)
