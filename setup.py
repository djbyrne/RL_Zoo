from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="RL_Zoo",
    version="0.0.1",
    author="Donal Byrne",
    author_email="donaljbyrne@me.com",
    description="Modular examples of RL algorithms",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/djbyrne/RL_Zoo",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)