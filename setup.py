from setuptools import setup, find_packages

setup(
    name="diff4sbi",
    packages=find_packages("tasks", "tasks.*"),
    install_requires=["sbi", "lampe", "zuko", "tueplots", "seaborn"],
)
