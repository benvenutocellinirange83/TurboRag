from setuptools import setup, find_packages

setup(
    name="turborag",
    version="1.0.0",
    description="Local quantized RAG engine — low CPU, low RAM, fully offline",
    packages=find_packages(exclude=["tests*", "sdk*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.0",
    ],
    extras_require={
        "llm":       ["llama-cpp-python>=0.2.90"],
        "langchain": ["langchain-core>=0.2", "langchain-community>=0.2"],
        "mcp":       ["fastmcp>=0.9"],
        "all": [
            "llama-cpp-python>=0.2.90",
            "langchain-core>=0.2",
            "langchain-community>=0.2",
            "fastmcp>=0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "turborag-api=turborag.api.server:main",
            "turborag-mcp=turborag.mcp.server:main",
        ],
    },
)
