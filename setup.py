from setuptools import setup

setup(
    name="int_fp_qsim",
    version="0.1.0",
    description="Lightmatter Flexible Precision (INT, FP) Simulator",
    maintainer="Lightmatter Machine Learning Team",
    maintainer_email="lakshmi@lightmatter.co, darius@lightmatter.co",
    url="https://github.com/lightmatter-ai/INT-FP-QSim",
    install_requires=[
        "pytorch-quantization==2.1.2",
        "transformers==4.27.4",
        "diffusers==0.16.1",
        "qtorch==0.3.0",
        "ninja"
    ],
    license="Apache 2.0",
)
