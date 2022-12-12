import numpy as np
from rob_sim.models import CartPole
from rob_sim.lib import FixedVal, NormDistr, UniformDistr
from rob_sim.lib import ModelFactory, Simulator, Logger
from rob_sim.lib import NNPolicy

import matplotlib.pyplot as plt

# policy
policy = NNPolicy("policies/cart_pole_sw_up")

# models factory
pend_params = {'masscart': UniformDistr(0.5,1.5),
                'masspole': UniformDistr(0.02,0.18),
                'length': FixedVal(0.5),
                'force_mag': FixedVal(10.0)}
pend_factory = ModelFactory(CartPole,pend_params)

# init position
X = np.array([0,0,np.pi,0.0])
# integration step
dt = UniformDistr(0.02,0.08)
#measurement noise
noise = NormDistr(0.0,0.05,shape=(4,))

# step number
n_step = 200
n_model = 50

# create model list
mdl_list = []
for i in range(n_model):
    mdl_list.append(Simulator(pend_factory.create(),dt,noise,X, Logger()))

# simulation
for i in range(n_step):
    for pend_sim in mdl_list:
        act = policy.get_action(pend_sim.get_observe())
        pend_sim.step(act)

# plot result
for pend_sim in mdl_list:
    state_log = pend_sim.get_log_state()
    plt.plot(state_log[0,:])
plt.show()

for pend_sim in mdl_list:
    state_log = pend_sim.get_log_state()
    plt.plot(state_log[1,:])
plt.show()

for pend_sim in mdl_list:
    state_log = pend_sim.get_log_state()
    plt.plot(state_log[2,:])
plt.show()

for pend_sim in mdl_list:
    state_log = pend_sim.get_log_state()
    plt.plot(state_log[3,:])
plt.show()