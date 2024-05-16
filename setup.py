from setuptools import setup
import unittest

from setuptools import find_packages
REQUIRED_PACKAGES = [
    'dm-haiku>0.0.9',
    'h5py',
    'hydra-core',
    'jax<0.4.24',
    'jaxlib<0.4.24',
    'jax-dataclasses',
    # TODO(b/230487443) - use released version of kfac.
    'kfac_jax @ git+https://github.com/deepmind/kfac-jax',
    'ml-collections',
    'pyscf',
    # 'tensorboard',
    'tensorboard',
    'pyyaml',
    'tqdm',
    'uncertainties',
    'scipy<1.13.0', 
    folx,
    jaxite,
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
