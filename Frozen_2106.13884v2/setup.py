from setuptools import setup, find_packages

setup(
    name="frozen_model",  # <-- Change this to "lora_model" or "frozen_model"
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)