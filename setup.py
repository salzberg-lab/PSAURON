from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()
    
setup(
    name="psauron",
    version="1.0.4",
    description="A tool to assess protein coding gene annotation",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Markusjsommer/psauron",
    author="markus",
    author_email = "markusjsommer@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    install_requires=["torch>=2.1.2",
                      "torchvision>=0.16.2",
                      "torchaudio>=2.1.2",
                      "typing-extensions>=4.9.0",
                      "tqdm>=4.66.1",
                      "scipy>=1.10.1",
                      "numpy>=1.24.4",
                      "pandas>=2.0.3",
                      "setuptools"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2", "pytest-cov>=4.0", "wheel"],
    },
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'psauron = psauron.psauron:eye_of_psauron',
        ],
    },
    include_package_data=True,
    package_data={'': ['/data/model_state_dict.pt']},
    
)