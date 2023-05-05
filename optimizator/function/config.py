
"""
Jeux de paramètres globaux, invariants sur les expériences
"""

import numpy as np
import jax.numpy as jnp
from scipy.linalg import eigh
from function.helper_functions import *

Nt = 4096
Nh = 20
IDFT_1ddl = jnp.array(construction_IDFT(Nt,Nh,1), dtype=jnp.float64)
DFT_1ddl = jnp.array(construction_DFT(Nt,Nh,1), dtype=jnp.float64)

kn = 1e4 #1e3
eps = 1e1 #5e1
g0 = 0.4

m2 = .5
m1 = 1
M = np.array([[m2,0],
			[0,m1]],dtype=np.float32)
k1 = 100
k2 = 300
K = np.array([[k1+k2,-k2],
			[-k2,k2]],dtype=np.float32)

xi = 0.01
seuil_xi = np.array([1])
fact_xi = np.array([1])
C = amort_modal(K,M,xi,seuil_xi,fact_xi)

f0 = 20
w,p = eigh(K,M)
F = f0*M@p[:,0]

beta=1
gamma=1
ddl_nl = np.array([1],dtype=np.int64)
ddl_ln = np.array([0],dtype=np.int64)

derivatives = False
penalite = None
nu = 4

omegas_range = [7.0/nu, 12.0/nu]
#omegas_range = [11.5/nu, 12.0/nu]
omegas_steps = 100#3