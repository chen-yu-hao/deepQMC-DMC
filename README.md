# deepQMC-DMC: connect deepQMC and DMC

A collection of GPU-friendly and neural-network-friendly scalable QMC implementations in JAX.

## Installation
deepQMC-DMC can be installed via the supplied setup.py file.
```shell
pip3 install -e .
```
You need to install jaqmc before using deepQMC-DMC. (https://github.com/bytedance/jaqmc)

## Train VMC wave function 
### Create a molecule
A Molecule can be also created from scratch by specifying the nuclear coordinates and charges, as well as the total charge and spin multiplicity:

```python
from deepqmc_dmc import Molecule
mol = Molecule(  # LiH
    coords=[[0.0, 0.0, 0.0], [3.015, 0.0, 0.0]],
    charges=[3, 1],
    charge=0,
    spin=0,
    unit='bohr',
)
```
### Create the molecular Hamiltonian
From the molecule the MolecularHamiltonian is constructed:
```python
from deepqmc_dmc import MolecularHamiltonian

H = MolecularHamiltonian(mol=mol)
```
### Create a wave function ansatz
```python
import os

import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import deepqmc_dmc
from deepqmc_dmc.app import instantiate_ansatz
from deepqmc_dmc.wf import NeuralNetworkWaveFunction


deepqmc_dir = os.path.dirname(deepqmc_dmc.__file__)
config_dir = os.path.join(deepqmc_dir, 'conf/ansatz')

with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='default')

_ansatz = instantiate(cfg, _recursive_=True, _convert_='all')

ansatz = instantiate_ansatz(H, _ansatz)
```

