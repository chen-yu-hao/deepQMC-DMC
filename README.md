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

```shell
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
```shell
from deepqmc import MolecularHamiltonian

H = MolecularHamiltonian(mol=mol)
```shell
