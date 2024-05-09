from deepqmc import Molecule
from deepqmc.sampling import chain, MetropolisSampler, DecorrSampler
from deepqmc import train
import os
import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import deepqmc
from deepqmc.app import instantiate_ansatz
from deepqmc.wf import NeuralNetworkWaveFunction
from deepqmc.types import PhysicalConfiguration
from deepqmc import MolecularHamiltonian
import jax
import jax.numpy as jnp
import pathlib
import sys
import time
import os
import jax.numpy as jnp
from absl import app
from absl import logging
# from lapnet import base_config
# from lapnet import checkpoint
# from lapnet import networks
# from lapnet import hamiltonian
import jax
import jax.numpy as jnp

from jaqmc.dmc import run
from jaqmc.dmc.ckpt_metric_manager import DataPath
from ml_collections.config_flags import config_flags
import ml_collections


def phys_conf(R, r, **kwargs):
    if r.ndim == 2:
        return PhysicalConfiguration(R, r, jnp.array(0))
    n_smpl = len(r)
    return PhysicalConfiguration(
        jnp.tile(R[None], (n_smpl, 1, 1)),
        r,
        jnp.zeros(n_smpl, dtype=jnp.int32),
    )

dmc_cfg = ml_collections.ConfigDict({
          'iterations': 10000,
          'time_step': 0.001,
          'update_energy_offset_interval': 1,
          'energy_offset_update_amplitude':1,
          'print_info_interval': 20,
          'weight_branch_threshold': (0.3, 2),
          'energy_window_size': 1000,
          'mixed_estimator_num_steps': 5000,
          'energy_outlier_rel_threshold': -1.0,
          'energy_cutoff_alpha': 0.2,
          # Whether do fix-size branching or not.
          # It can be turned on to boost efficiency due to better JAX jitting.
          'fix_size': False,
          # If True, use elec-by-elec moves rather than walker-by-walker moves.
          # By default it's turned off due to efficiency concern.
          'ebye_move': False,
          # Negative `effective_time_step_update_period` means always
          # update effective time step.
          'effective_time_step_update_period': -1,

          # The size of a block of iterations. The recovery mechanism will
          # roll back to the previous block when error happens.
          'block_size': 5000,
          # The max number of rolling-back which the recovery mechasim will
          # perform before it gives up and abort the process.
          'max_restore_nums': 3,

          'log': {
              # The local path that the checkpoint will be saved to.
              'save_path': '',
              # The remote path that the checkpoint will be upload to.
              'remote_save_path': '',
              # The local path that the previous checkpoint will be loaded from.
              'restore_path': '',
              # The remote path that the previous checkpoint will be downloaded from.
              'remote_restore_path': '',
          },
      }
  )

class deepqmc_wave():
    def __init__(self,mol,ansatz,H):
        self.mol=mol
        self.H=H
        self.ansatz=ansatz
        self.R=mol.coords
    def load_deepqmc_model(self,checkpoint_path):
        from deepqmc.types import PhysicalConfiguration
        step, train_state = jnp.load(checkpoint_path,allow_pickle=True)
        from functools import partial
        for i in train_state[1]:
            for j in train_state[1][i]:
                train_state[1][i][j]=train_state[1][i][j].reshape((train_state[1][i][j].shape[1:]))
        params = train_state[1]
        wf=partial(self.ansatz.apply,params)
        self.wf=jax.vmap(wf)
        self.r =train_state[0]['r']
        self.params=params
        self.data=train_state
        self.local_energy = jax.vmap(self.H.local_energy(partial(self.ansatz.apply, params)))
    def deepqmc_psi(self,r):
        # print(r)
        r=r.reshape([1,r.shape[0]//3,3])
        out=self.wf(phys_conf(self.R,r))
        return (out[0][0]).astype(jnp.float32),(out[1][0]).astype(jnp.float32)
    def deepqmc_energy(self,r):
        # print(r)
        r=r.reshape([1,r.shape[0]//3,3])
        return (self.local_energy(None,phys_conf(self.R,r))[0][0]).astype(jnp.float32)
        
        