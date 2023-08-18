from setuptools import find_packages, setup

requirements = [
    "nfl_data_py",
    "scikit-learn",
    "numpy",
    "matplotlib",
    "pandas",
    "seaborn",
    "xgboost",
    "bayesian-optimization"
]

setup(
    name="nfl_wp_model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=requirements
)
