"""Setup configuration for CryptoShift package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cryptoshift",
    version="1.0.0",
    author="CryptoShift Team",
    description="Real-time cryptocurrency market anomaly detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cryptoshift/cryptoshift",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "APScheduler>=3.10.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "pycoingecko>=3.1.0",
        "requests>=2.31.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "httpx>=0.24.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "cryptoshift-download=scripts.download_historical:main",
            "cryptoshift-backtest=scripts.run_backtest:main",
            "cryptoshift-train=scripts.train:main",
            "cryptoshift-ingest=scripts.start_ingestion:main",
        ],
    },
)
