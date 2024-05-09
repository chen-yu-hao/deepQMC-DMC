import os
import haiku as hk
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import jax
import jax.numpy as jnp
import jax
from jaqmc.dmc import run 
import ml_collections
import time

import logging
import os
import sys

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

if not os.environ.get('NVIDIA_TF32_OVERRIDE'):
    # disable TensorFloat-32 for better precision
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    if 'jax' in sys.modules:
        log.warning(
            'JAX was imported before deepqmc, TensorFloat32 precision might be enabled.'
            ' You may experience numerical issues.'
        )
elif os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
    log.warning(
        'TensorFloat-32 seems to be enabled. You might want to disable TensorFloat-32'
        ' precision by setting NVIDIA_TF32_OVERRIDE=0 before loading deepqmc to avoid'
        ' numerical issues.'
    )


import jax  # noqa: E402

from .conf.custom_resolvers import get_hydra_subdir  # noqa: E402
from .hamil import MolecularHamiltonian  # noqa: E402
from .molecule import Molecule  # noqa: E402
from .sampling import (  # noqa: E402
    DecorrSampler,
    MetropolisSampler,
    ResampledSampler,
)
from .train import train  # noqa: E402

jax.config.update('jax_default_matmul_precision', 'highest')
OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('get_hydra_subdir', get_hydra_subdir)

__all__ = [
    'DecorrSampler',
    'MetropolisSampler',
    'MolecularHamiltonian',
    'Molecule',
    'ResampledSampler',
    'train',
]



def phys_conf(R, r, **kwargs):
    if r.ndim == 2:
        return PhysicalConfiguration(R, r, jnp.array(0))
    n_smpl = len(r)
    return PhysicalConfiguration(
        jnp.tile(R[None], (n_smpl, 1, 1)),
        r,
        jnp.zeros(n_smpl, dtype=jnp.int32),
    )



class ml_wf():
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

def dmc_run(dmc_cfg,wf):
    position_shape=wf.r[0,0].shape
    position = wf.r[0,0].reshape(position_shape[0],position_shape[1]*position_shape[2])
    # dmc_cfg=deepqmc_psi.
    key = jax.random.PRNGKey(int(time.time()))
    run(
        position,
        dmc_cfg.iterations,
        wf.deepqmc_psi,
        dmc_cfg.time_step, key,
        nuclei=jnp.array(wf.mol.coords),
        charges=jnp.array(wf.mol.charges),

        # Below are optional arguments
        mixed_estimator_num_steps=dmc_cfg.mixed_estimator_num_steps,
        energy_window_size=dmc_cfg.energy_window_size,
        weight_branch_threshold=dmc_cfg.weight_branch_threshold,
        update_energy_offset_interval=dmc_cfg.update_energy_offset_interval,
        energy_offset_update_amplitude=dmc_cfg.energy_offset_update_amplitude,
        energy_cutoff_alpha=dmc_cfg.energy_cutoff_alpha,
        effective_time_step_update_period=dmc_cfg.effective_time_step_update_period,
        energy_outlier_rel_threshold=dmc_cfg.energy_outlier_rel_threshold,
        fix_size=dmc_cfg.fix_size,
        ebye_move=dmc_cfg.ebye_move,
        block_size=dmc_cfg.block_size,
        max_restore_nums=dmc_cfg.max_restore_nums,
        save_path=dmc_cfg.log.save_path,
        restore_path=dmc_cfg.log.save_path,
        local_energy_func=wf.deepqmc_energy,
    )
    