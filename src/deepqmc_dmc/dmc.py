import ml_collections
from jaqmc.dmc import run 
import jax
import time
import jax.numpy as jnp

dmc_cfg = ml_collections.ConfigDict({
          'iterations': 10000,
          'time_step': 0.001,
          'update_energy_offset_interval': 1,
          'energy_offset_update_amplitude':1,
          'print_info_interval': 20,
          'weight_branch_threshold': (0.3, 2),
          'energy_window_size': 10000,
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

def dmc_run(dmc_cfg,wf,num_walker,batch_size):
    if wf.r.shape[0]==1:
        position_shape=wf.r[0,0].shape
        position = wf.r[0,0].reshape(position_shape[0],position_shape[1]*position_shape[2])
    else:
        position_shape=wf.r[0,0].shape
        position = jnp.concatenate([i[0] for i in wf.r],axis=0).reshape(position_shape[0]*wf.r.shape[0],position_shape[1]*position_shape[2])
    #position_shape=wf.r[0,0].shapei
    position = position[:num_walker]

    # print(position.shape)
    # position_shape=wf.r[0,0].shape
    # position = wf.r[0,0].reshape(position_shape[0],position_shape[1]*position_shape[2])
    # dmc_cfg=deepqmc_psi.
    # position_shape=wf.r[0,0].shape
    # position = wf.r[0,0].reshape(position_shape[0],position_shape[1]*position_shape[2])
    # dmc_cfg=deepqmc_psi.
    key = wf.rng
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
