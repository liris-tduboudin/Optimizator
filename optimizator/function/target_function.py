import torch

from function.helper_functions import fnl_tilde_rlhbm
from function.config import *

def target_function(hparams_dict, Xh):
    fnl_tilde = fnl_tilde_rlhbm(Xh,hparams_dict['IDFT_1ddl'],hparams_dict['DFT_1ddl'],g0,kn,eps,Nt)
    return torch.mm(hparams_dict['Z'], Xh.t()).t() + fnl_tilde - hparams_dict['Fh'].unsqueeze(0)