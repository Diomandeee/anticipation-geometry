from setuptools import setup, find_packages

setup(
    name="anticipation-geometry",
    version="0.1.0",
    description="Generalized anticipation geometry for conversational and motion intelligence",
    author="Mohamed Diomande",
    author_email="mo@openclaw.com",
    url="https://github.com/Diomandeee/anticipation-geometry",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "embeddings": ["sentence-transformers>=2.2.0"],
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
