import jax.numpy as jnp
import jax
from function.helper_functions import fnl_tilde_rlhbm
from function.config import *

def target_function_maker(hparams_dict):
    def target_function(Xh):
        fnl_tilde = fnl_tilde_rlhbm(Xh,IDFT_1ddl,DFT_1ddl,g0,kn,eps, Nt)
        return hparams_dict['Z']@Xh + fnl_tilde - hparams_dict['Fh']
    return target_function