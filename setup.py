from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8-sig") as f:
    requirements = f.readlines()

with open("LICENSE", encoding="utf-8-sig") as f:
    license = f.readlines()

setup(
    name='search-optimization-tools',
    version='0.0.1',
    author='Alaa Khamis and Yinan Wang',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="yinanwang@sierra.one",
    install_requires=requirements,
    packages=find_packages(exclude=("tests",)),
    license="MIT",
    python_requires=">=3.8,<3.9"
)