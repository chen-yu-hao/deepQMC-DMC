from setuptools import setup
import unittest

from setuptools import find_packages
REQUIRED_PACKAGES = [
    'h5py',
    'jax',
    'jaxlib',
    # TODO(b/230487443) - use released version of kfac.
    'kfac_jax @ git+https://github.com/deepmind/kfac-jax',
    'ml-collections',
]

setup(
    name="deepqmc-dmc",
    version="0.1",
    author="Yuhao Chen",
    author_email="chenyuhao@shu.edu.cn",
    description="",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)

if __name__ == "__main__":
    setup()
