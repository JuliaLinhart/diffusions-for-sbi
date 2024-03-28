from setuptools import find_packages, setup

setup(
    name="diff4sbi",
    packages=find_packages("tasks", "tasks.*"),
    install_requires=[
        "sbibm",
        "sbi",
        "lampe",
        "zuko",
        "numpyro",
        "POT",
        "tueplots",
        "seaborn",
    ],
)
