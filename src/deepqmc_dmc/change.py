import jax.numpy as jnp
import kfac_jax
from deepqmc_dmc.types import Psi
def change_train_state(filepath, num_gpu):
    step,train_state=jnp.load(filepath, allow_pickle=True)
    # step,train_state=jnp.load("./chkpt-50501.pt", allow_pickle=True)
    # for i in range(len(train_state)):
    i=1
    for j in train_state[i]:
        for k in train_state[i][j]:
            # print(i,j,k,end=" :  ")
            # if j == 'psi':
            #     train_state[i][j][0] = train_state[i][j][0].reshape([num_gpu, 1, -1])
            #     train_state[i][j][1] = train_state[i][j][1].reshape([num_gpu, 1, -1])
            # else:
            # print(train_state[i][j][k].shape)
            train_state[i][j][k] = train_state[i][j][k][:num_gpu]
            # print(train_state[i][j][k].shape)
    i=0
    for j in train_state[i]:
        if j == 'psi':
            train_state[i][j] = Psi(train_state[i][j][0].reshape([num_gpu, 1, -1]),train_state[i][j][1].reshape([num_gpu, 1, -1]))
            #     train_state[i][j][1] = 
            # pass
        elif j == 'r':
            print(j,train_state[i][j].shape)
            shape=train_state[i][j].shape
            train_state[i][j] = train_state[i][j].reshape([num_gpu, shape[1], -1, shape[3],shape[4]])
            print(j,train_state[i][j].shape)
        elif j=="tau":
            print(j,train_state[i][j].shape)
            train_state[i][j] = train_state[i][j][:num_gpu]
            print(j,train_state[i][j].shape)
        else:
            print(j)
            print(train_state[i][j].shape)
            train_state[i][j] = train_state[i][j].reshape([num_gpu, 1, -1])
            print(train_state[i][j].shape)
    i=2
    for j in train_state[i].velocities:
        for k in train_state[i].velocities[j]:
            train_state[i].velocities[j][k] = train_state[i].velocities[j][k][:num_gpu]
    for j in range(len(train_state[i].estimator_state.blocks_states)):
    # for j in [0,5]
        if type(train_state[i].estimator_state.blocks_states[j])==kfac_jax.curvature_blocks.KroneckerFactored.State:
            for k in range(len(train_state[i].estimator_state.blocks_states[j].factors)):
                # print(train_state[i].estimator_state.blocks_states[j].factors[k].raw_value.shape)
                train_state[i].estimator_state.blocks_states[j].factors[k].weight = train_state[i].estimator_state.blocks_states[j].factors[k].weight[:num_gpu]
                train_state[i].estimator_state.blocks_states[j].factors[k].raw_value = train_state[i].estimator_state.blocks_states[j].factors[k].raw_value[:num_gpu]

        else:
            train_state[i].estimator_state.blocks_states[j].diagonal_factors[0].weight = train_state[i].estimator_state.blocks_states[j].diagonal_factors[0].weight[:num_gpu]
            train_state[i].estimator_state.blocks_states[j].diagonal_factors[0].raw_value = train_state[i].estimator_state.blocks_states[j].diagonal_factors[0].raw_value[:num_gpu]
    train_state[2].data_seen=train_state[2].data_seen[:num_gpu]
    train_state[2].step_counter=train_state[2].step_counter[:num_gpu]
    # print(train_state[2].step_counter.shape)
    # print(train_state[i].estimator_state.blocks_states[j].factors[k].raw_value.shape)

    
    
    # train_state=train_state[0],train_state[1],None
    # print(train_state[2].estimator_state)
    # print(train_state[1].psi.shape)

    return step,train_state