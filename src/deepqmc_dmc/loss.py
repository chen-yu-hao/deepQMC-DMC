from functools import partial

import jax
import kfac_jax

from deepqmc.clip import median_log_squeeze_and_mask
from deepqmc.parallel import all_device_mean
from deepqmc.utils import masked_mean
import jax.numpy as jnp
from jax.lax import fori_loop
from jaxite.jaxite_lib import jax_helpers
from .ml_wf import *
# from folx import batched_vmap

__all__ = ()

def compute_local_energy(rng, hamil, ansatz, params, phys_conf,batch_size):
    rng = jax.random.split(rng, len(phys_conf))
    rng = jax.vmap(partial(jax.random.split, num=phys_conf.batch_shape[1]))(rng)
    
    func_vmap=hamil.local_energy(partial(ansatz.apply, params))
    local_energy,hamil_stats=jax.vmap(jax_helpers.batch_vmap(func_vmap, batch_size=batch_size))(rng,phys_conf)
    local_energy,hamil_stats=jax.vmap(jax_helpers.batch_vmap(func_vmap, batch_size=batch_size))(rng,phys_conf)
    # print(local_energy)
    hamil_stats = {
                'hamil/V_el': hamil_stats [:,:,0],
                'hamil/E_kin': hamil_stats[:,:,1],
                'hamil/V_loc': hamil_stats[:,:,2],
                'hamil/V_nl': hamil_stats [:,:,3],
                'hamil/lap': hamil_stats  [:,:,4],
                'hamil/quantum_force': hamil_stats[:,:,5],
            }

    stats = {
        'E_loc/mean': local_energy.mean(axis=1),
        'E_loc/std': local_energy.std(axis=1),
        'E_loc/min': local_energy.min(axis=1),
        'E_loc/max': local_energy.max(axis=1),
        **{k_hamil: v_hamil.mean(axis=1) for k_hamil, v_hamil in hamil_stats.items()},
    }
    return local_energy, stats


def clip_local_energy(clip_mask_fn, local_energy):
    return jax.vmap(clip_mask_fn)(local_energy)

def compute_spin_plus(ansatz, phys_conf, params):

    flat_phys_conf = phys_conf
    print(flat_phys_conf.r)
    print(flat_phys_conf.R)
    vspsi=jax.vmap(partial(ansatz.apply, params))

    sign0,psi0=vspsi(flat_phys_conf)
    S2_local=jnp.zeros_like(psi0)

    n_up=4

    for i in range(n_up):
        # for j in range(n_down):
        flat_phys_conf1=flat_phys_conf
        flat_phys_conf1.r=flat_phys_conf1.r.at[:,i,:].set(flat_phys_conf.r[:,n_up,:])
        flat_phys_conf1.r=flat_phys_conf1.r.at[:,n_up,:].set(flat_phys_conf.r[:,i,:])
        sign1,psi1=vspsi(flat_phys_conf1)
        S2_local=S2_local+sign0*sign1*jnp.exp(psi1-psi0)
    return S2_local

def compute_spin_plus_tangent(ansatz, phys_conf, params, params_tangent):
    def compute_spin(params):
        return compute_spin_plus(ansatz, phys_conf,params)
    spin_plus,spin_plus_tangent = jax.jvp(compute_spin, (params,), (params_tangent,))
    return spin_plus,spin_plus_tangent

def compute_log_psi_tangent(ansatz, phys_conf, params, params_tangent):
    flat_phys_conf = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), phys_conf
    )

    def flat_log_psi(params):
        return jax.vmap(ansatz.apply, (None, 0))(params, flat_phys_conf).log

    log_psi, log_psi_tangent = jax.jvp(flat_log_psi, (params,), (params_tangent,))
    kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
    return log_psi_tangent.reshape(phys_conf.mol_idx.shape)


def compute_mean_energy(local_energy, weight):
    return all_device_mean(local_energy * weight)


def compute_mean_energy_tangent(local_energy, weight, log_psi_tangent, gradient_mask):
    per_mol_mean_energy = all_device_mean(local_energy * weight, axis=-1, keepdims=True)
    local_energy_tangent = (
        (local_energy - per_mol_mean_energy) * log_psi_tangent * weight
    )
    mean_energy_tangent = masked_mean(local_energy_tangent, gradient_mask)
    return mean_energy_tangent


def create_energy_loss_fn(hamil, ansatz, clip_mask_fn,batch_size):
    @jax.custom_jvp
    def loss_fn(params, rng, batch):
        phys_conf, weight = batch
        local_energy, stats = compute_local_energy(
            rng, hamil, ansatz, params, phys_conf,batch_size
        )
        # rng,rng1=jnp.random.split(rng,1)
        # local_energy_batch, stats = compute_local_energy(
        #     rng, hamil, ansatz, params, phys_conf[:,500:1000]
        # )
        # local_energy = jnp.concatenate((local_energy, local_energy_batch),axis=1)
        # local_energy = jnp.concatenate((local_energy, local_energy_batch),axis=1)
        # local_energy = jnp.concatenate((local_energy, local_energy_batch),axis=1)
        loss = compute_mean_energy(local_energy, weight)
        ################# for spin plus ###########################
        spin_plus= compute_spin_plus(ansatz, phys_conf,params)
        loss_spin_plus = compute_mean_energy(spin_plus, weight)
        stats["loss_spin_plus"] = loss_spin_plus
        stats["loss"] = loss
        ###########################################################

        return loss+loss_spin_plus*10, (local_energy, stats)
        # return loss, (local_energy, stats)

    @loss_fn.defjvp
    def loss_fn_jvp(primals, tangents):
        params, rng, (phys_conf, weight) = primals
        params_tangent, *_ = tangents

        local_energy, stats = compute_local_energy(
            rng, hamil, ansatz, params, phys_conf,batch_size
        )

        loss = compute_mean_energy(local_energy, weight)

        log_psi_tangent = compute_log_psi_tangent(
            ansatz, phys_conf, params, params_tangent
        )
        clipped_local_energy, gradient_mask = clip_local_energy(
            clip_mask_fn or median_log_squeeze_and_mask, local_energy
        )
        loss_tangent = compute_mean_energy_tangent(
            clipped_local_energy, weight, log_psi_tangent, gradient_mask
        )

        ##################### spin plus #################################
        spin_plus,spin_plus_tangent=compute_spin_plus_tangent(ansatz, phys_conf, params, params_tangent)
        loss_spin_plus = compute_mean_energy(spin_plus, weight)
        loss_spin_plus_tangent = 2 * loss_spin_plus * compute_mean_energy(2*(spin_plus-loss_spin_plus)*log_psi_tangent+spin_plus_tangent, weight)


        #################################################################

        aux = (local_energy, stats)
        return (loss+10*loss_spin_plus**2, aux), (loss_tangent+10*loss_spin_plus_tangent, aux)
        # return (loss, aux), (loss_tangent, aux)


        # jax.custom_jvp has actually no official support for auxiliary output.
        # the second aux in the tangent output should be in fact aux_tangent.
        # we just output the same thing to satisfy jax's API requirement with
        # the understanding that we'll never need aux_tangent

    return loss_fn
