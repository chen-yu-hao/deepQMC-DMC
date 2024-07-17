from functools import partial

import jax
import kfac_jax
from .types import PhysicalConfiguration
from .clip import median_log_squeeze_and_mask
from .parallel import all_device_mean
from .utils import masked_mean
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

def phys_conf0(R, r, **kwargs):
    if r.ndim == 3:
        return PhysicalConfiguration(R, r, jnp.array(0))
    n_smpl = len(r)
    return PhysicalConfiguration(
    jnp.tile(R[None], (n_smpl, 1, 1)),
            r,
            jnp.zeros(n_smpl, dtype=jnp.int32),
            )

def compute_log_psi_tangent(ansatz, phys_conf, params, params_tangent):
    flat_phys_conf = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), phys_conf
    )
 
    def flat_log_psi(params):
        return jax.vmap(ansatz.apply, (None, 0))(params, flat_phys_conf).log

    log_psi, log_psi_tangent = jax.jvp(flat_log_psi, (params,), (params_tangent,))
    kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
    return log_psi_tangent.reshape(phys_conf.mol_idx.shape)

def compute_spin_plus(rand,ansatz,hamil, phys_conf, params):
    nele=phys_conf.r.shape[2]
    flat_phys_conf = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), phys_conf
                )
    # print(flat_phys_conf.r)
    # print(flat_phys_conf.R)
    # print(flat_phys_conf)
    vspsi =jax.vmap(partial(ansatz.apply, params))
    #vspsi=jax.vmap(jax.vmap(partial(ansatz.apply, params)))
    sign0,psi0=vspsi(flat_phys_conf)
    S2_local=jnp.zeros_like(psi0)

    n_up=(nele+hamil.mol.spin)//2
    n_down=nele-n_up
    print(psi0)
    sign=jnp.zeros((n_up,*psi0.shape))
    psi=jnp.zeros((n_up,*psi0.shape))

    for i in range(n_up):
        # for j in range(n_down):
        r=flat_phys_conf.r.reshape(-1,nele,3)
        r=r.at[:,i,:].set(flat_phys_conf.r[:,n_up+rand,:])
        r=r.at[:,n_up+rand,:].set(flat_phys_conf.r[:,i,:])
        #r=r.reshape(-1,nele*3)
        flat_phys_conf0=PhysicalConfiguration(flat_phys_conf.R,r,flat_phys_conf.mol_idx)
        # print(flat_phys_conf0.R,flat_phys_conf0.r)
        sign1,psi1=vspsi(flat_phys_conf0)
        #print(flat_phys_conf1.R,flat_phys_conf1.r)
        S2_local=S2_local+sign0*sign1*jnp.exp(psi1-psi0)
        sign=sign.at[i].set(sign1)
        psi=psi.at[i].set(psi1)
    return (1-S2_local),(sign0,psi0),(sign,psi)



# def compute_spin_plus_tangent(rng,ansatz,hamil, phys_conf, params, params_tangent):
#     def compute_spin(params):
#         return compute_spin_plus(rng,ansatz,hamil, phys_conf,params)
#     spin_plus,spin_plus_tangent = jax.jvp(compute_spin, (params,), (params_tangent,))
#     return spin_plus,spin_plus_tangent

def compute_log_psi_tangent_i(ansatz, phys_conf, params, params_tangent):
    flat_phys_conf = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), phys_conf
    )
 
    def flat_log_psi(params):
        return jax.vmap(ansatz.apply, (None, 0))(params, flat_phys_conf).log

    log_psi, log_psi_tangent = jax.jvp(flat_log_psi, (params,), (params_tangent,))
    # kfac_jax.register_normal_predictive_distribution(log_psi[:, None])
    return log_psi_tangent.reshape(phys_conf.mol_idx.shape)

def compute_i_tangent(sign0,psi0,ansatz,params,params_tangent,R,mol_idx,sign,psi,r):

    flat_phys_conf0=PhysicalConfiguration(R,r,mol_idx)
    psi_tangent = compute_log_psi_tangent_i(
        ansatz, flat_phys_conf0, params, params_tangent
    )
    spin_plus_tangent=-(sign0*sign)*psi_tangent*jnp.exp(psi-psi0)
    return spin_plus_tangent

def compute_spin_plus_tangent(rng,ansatz,hamil, phys_conf, params, params_tangent):

    nele=phys_conf.r.shape[2]
    n_up=(nele+hamil.mol.spin)//2
    n_down=nele-n_up
    rand=jax.random.randint(rng, (1,), 0, n_down)
    rand=rand[0]
    spin_plus,(sign0,psi0),(sign,psi)=compute_spin_plus(rand,ansatz,hamil, phys_conf,params)
    
    # compute_tangent=partial(compute_i_tangent,sign0,psi0,ansatz,params,params_tangent,phys_conf.R,phys_conf.mol_idx)
    # vmapped_compute_i_tangent=jax_helpers.batch_vmap(compute_tangent,batch_size=1)
    # vampped_r=jnp.zeros([n_up,*phys_conf.r.shape])
    # # vampped_R=jnp.zeros([n_up,*phys_conf.R.shape])
    # # vampped_mol_idx=jnp.zeros([n_up,*phys_conf.mol_idx.shape])
    # for i in range(n_up):
    #     vampped_r=vampped_r.at[i].set(phys_conf.r)
    #     vampped_r=vampped_r.at[i,:,:,i,:].set(phys_conf.r[:,:,n_up+rand,:])
    #     vampped_r=vampped_r.at[i,:,:,n_up+rand,:].set(phys_conf.r[:,:,i,:])
    #     # vampped_R=vampped_R.at[i].set(phys_conf.R)
    #     # vampped_mol_idx=vampped_mol_idx.at[i].set(phys_conf.mol_idx)
    # vmapped_spin_plus_tangent=vmapped_compute_i_tangent(sign,psi,vampped_r)
    # spin_plus_tangent=jnp.sum(vmapped_spin_plus_tangent,axis=0)
    i=0
    r=phys_conf.r
    r=r.at[:,:,i,:].set(phys_conf.r[:,:,n_up+rand,:])
    r=r.at[:,:,n_up+rand,:].set(phys_conf.r[:,:,i,:])

    flat_phys_conf0=PhysicalConfiguration(phys_conf.R,r,phys_conf.mol_idx)
    psi_tangent = compute_log_psi_tangent_i(
            ansatz, flat_phys_conf0, params, params_tangent
        )
    spin_plus_tangent=-(sign0*sign[i])*psi_tangent*jnp.exp(psi[i]-psi0)

    for i in range(1,n_up):
        r=phys_conf.r
        r=r.at[:,:,i,:].set(phys_conf.r[:,:,n_up+rand,:])
        r=r.at[:,:,n_up+rand,:].set(phys_conf.r[:,:,i,:])

        flat_phys_conf0=PhysicalConfiguration(phys_conf.R,r,phys_conf.mol_idx)
        psi_tangent = compute_log_psi_tangent_i(
            ansatz, flat_phys_conf0, params, params_tangent
        )
        spin_plus_tangent=spin_plus_tangent-(sign0*sign[i])*psi_tangent*jnp.exp(psi[i]-psi0)
    
    return spin_plus,spin_plus_tangent



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
        ## rng,rng1=jnp.random.split(rng,1)
        ## local_energy_batch, stats = compute_local_energy(
        ##     rng, hamil, ansatz, params, phys_conf[:,500:1000]
        ## )
        ## local_energy = jnp.concatenate((local_energy, local_energy_batch),axis=1)
        ## local_energy = jnp.concatenate((local_energy, local_energy_batch),axis=1)
        ## local_energy = jnp.concatenate((local_energy, local_energy_batch),axis=1)
        loss = compute_mean_energy(local_energy, weight)
        ################# for spin plus ###########################
        spin_plus= compute_spin_plus(ansatz, hamil,phys_conf,params)
        loss_spin_plus = compute_mean_energy(spin_plus, weight)
        stats["loss_spin_plus"] = loss_spin_plus
        stats["loss"] = loss
        ###########################################################
        #loss=0
        #local_energy=0
        #stats=0
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
        # print(loss_tangent)

        #################################################################   std   ###################################################################################################
        loss_std= jnp.sqrt(compute_mean_energy(local_energy**2, weight)-loss**2) # \sqrt{<H2>-<H>2}

        clipped_local_energy2, gradient_mask = clip_local_energy(
            clip_mask_fn or median_log_squeeze_and_mask, local_energy**2
        )
        H2_tangent = compute_mean_energy_tangent(
            clipped_local_energy2, weight, log_psi_tangent, gradient_mask
        )

        loss_std_tangent =  (H2_tangent-2*loss*loss_tangent) / loss_std
        stats['loss_std'] = jnp.mean(loss_std)
        # print(loss_std_tangent)
        #############################################################################################################################################################################
        ############################################################### spin plus ###################################################################################################
        spin_plus,spin_plus_tangent=compute_spin_plus_tangent(rng,ansatz, hamil, phys_conf, params, params_tangent)
        loss_spin_plus = compute_mean_energy(spin_plus, weight)
        loss_spin_plus_tangent = 2 * loss_spin_plus * compute_mean_energy((spin_plus-2*loss_spin_plus+1)*log_psi_tangent+spin_plus_tangent, weight)
        stats["loss_spin_plus"] = loss_spin_plus
        stats["loss"] = loss
        stats['S2'] = jnp.mean(spin_plus)
        nele=phys_conf.r.shape[-2]
        n_up=(nele+hamil.mol.spin)//2
        n_down=nele-n_up
        # omega_spin=hamil.omega_spin
        #############################################################################################################################################################################



        # omega_loss,omega_std,omega_spin=0.0,0.0,1.0  # for testing###################################################################################################################
        omega_loss,omega_std,omega_spin=hamil.omega_loss,hamil.omega_std,hamil.omega_spin

        aux = (local_energy, stats)

        # print(loss_tangent,loss_spin_plus_tangent,loss_std_tangent,log_psi_tangent)

        return (omega_loss*loss+omega_spin*n_down**2*loss_spin_plus**2+omega_std*loss_std, aux), (omega_loss*loss_tangent+omega_spin*n_down**2*loss_spin_plus_tangent+omega_std*loss_std_tangent, aux)
        # return (loss, aux), (loss_tangent, aux)


        # jax.custom_jvp has actually no official support for auxiliary output.
        # the second aux in the tangent output should be in fact aux_tangent.
        # we just output the same thing to satisfy jax's API requirement with
        # the understanding that we'll never need aux_tangent

    return loss_fn
