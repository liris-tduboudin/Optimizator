import torch

from function.helper_functions import fnl_tilde_rlhbm
from function.config import *

def target_function(hparams_dict, Xh, device):

    fnl_tilde = fnl_tilde_rlhbm(Xh,IDFT_1ddl_tensor,DFT_1ddl_tensor,g0,kn,eps, Nt, device)

    return torch.mm(hparams_dict['Z'], Xh.t()).t() + fnl_tilde - hparams_dict['Fh'].unsqueeze(0)