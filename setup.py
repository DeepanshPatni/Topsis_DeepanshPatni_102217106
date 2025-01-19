from setuptools import setup, find_packages

setup(
    name="102217106_topsis",
    version="1.0.0",
    description="A simple TOPSIS implementation package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Deepansh Patni",
    author_email="deepanshpatni@gmail.com",
    packages=find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
