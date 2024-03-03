from setuptools import find_packages, setup

setup(
    name="diff4sbi",
    packages=find_packages(),
    install_requires=["sbi", "lampe", "zuko", "tueplots", "seaborn"],
)
