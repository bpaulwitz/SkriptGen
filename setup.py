from sys import version
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['einops', 'opencv-python', 'pandas', 'torch', 'torchvision']

setup(
    name='ScriptGen',
    version='0.1',
    url='https://github.com/bpaulwitz/ScriptGen',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any']
)