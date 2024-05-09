import jax.numpy as jnp
import jax
from .types import PhysicalConfiguration


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
    def __init__(self,mol,ansatz,H,seed):
        self.mol=mol
        self.H=H
        self.ansatz=ansatz
        self.R=mol.coords
        self.rng=jax.random.PRNGKey(seed)
    def load_deepqmc_model(self,checkpoint_path):
        from deepqmc.types import PhysicalConfiguration
        step, train_state = jnp.load(checkpoint_path,allow_pickle=True)
        self.data=train_state
        from functools import partial
        for i in train_state[1]:
            for j in train_state[1][i]:
                train_state[1][i][j]=train_state[1][i][j][0]
        params = train_state[1]
        wf=partial(self.ansatz.apply,params)
        self.wf=jax.vmap(wf)
        self.r =train_state[0]['r']
        self.params=params
        self.local_energy = jax.vmap(self.H.local_energy(partial(self.ansatz.apply, params)))
    def deepqmc_psi(self,r):
        # print(r)
        r=r.reshape([1,r.shape[0]//3,3])
        out=self.wf(phys_conf(self.R,r))
        return (out[0][0]),(out[1][0])
    def deepqmc_energy(self,r):
        # print(r)
        r=r.reshape([1,r.shape[0]//3,3])
        return (self.local_energy(jax.random.split(self.rng,1),phys_conf(self.R,r))[0][0])