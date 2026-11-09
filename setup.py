from setuptools import find_packages, setup

setup(
    name="it_cuda_been_faster",
    version="0.1.0",
    description="Automating GPU kernel optimization using reinforcement learning.",
    author="Anna Thomas",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "trl>=0.15.0",
        "ninja>=1.11.0",
        "pybind11>=2.12.0",
        "accelerate>=0.30.0",
        "datasets>=2.18.0",
        "scipy>=1.10.0",
    ],
)
