from setuptools import find_packages, setup

setup(
    name="bert-optimization",
    version="0.0.1",
    packages=find_packages(exclude=["tests"]),
    install_requires=["tensorflow", "tensorflow-addons"],
    author="Jeong Ukjae",
    author_email="jeongukjae@gmail.com",
    url="https://github.com/jeongukjae/bet-optimization",
    description="",
)
