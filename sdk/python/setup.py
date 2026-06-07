from setuptools import setup, find_packages

setup(
    name="turborag-sdk",
    version="1.0.0",
    description="Python client SDK for the TurboRag REST API",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],   # zero dependencies — stdlib only
)
