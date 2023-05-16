import numpy as np
import torch
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import sys
import pickle as pkl

sys.path.insert(0, '..')
from function.helper_functions import *
from function.config import *

np.set_printoptions(threshold=sys.maxsize,linewidth=200,precision=2)

colors_poly = {'red_poly':(185/255,30/255,50/255),'orange_poly':(250/255,150/255,30/255),
               'gray_poly':(166/255,168/255,171/255),'green_poly':(140/255,200/255,60/255),
               'blue_poly':(65/255,170/255,230/255)}

if __name__=='__main__':

    """"
    Calcul de la FRF linéaire pour exemple d'appel de la fonction créant Z et Fh (efforts exterieurs)
    """
    n_pts = 1000
    omegas = np.linspace(7,12,n_pts)
    amps_lin = np.zeros_like(omegas)
    for i in range(n_pts):
        omega = omegas[i]
        Z,Fh = construction_Zred_Fred_dZreddw_dFreddw(omega,M,C,K,F,Nh,beta,gamma,ddl_ln,ddl_nl,derivatives,penalite,nu)
        x_tilde = np.linalg.solve(Z,-Fh)
        x_t = IDFT_1ddl@x_tilde
        amps_lin[i] = np.max(np.abs(x_t))

    """
    Chargement d'un résultat non-linéaire pour la même valeur de F
    """
    pkl_file = './crf_file.pkl'
    with open(pkl_file,'rb') as file:
        res_dict = pkl.load(file)
    amps_RL = res_dict['amplitudes']
    omegas_RL = res_dict['omegas']
    Xhs_RL = res_dict['Xhs_full']


    """"
    Vérification que la fonction pytorch donne le bon effort => OK
    """
    idx_RL = 800
    Xhs_RL_idx = Xhs_RL[idx_RL,1::2]
    omega_idx = omegas_RL[idx_RL]
    fig,(ax_dep,ax_eff) = plt.subplots(2,1)
    t = np.linspace(0,2*np.pi/omega_idx*(1-1/Nt),Nt)
    Z,Fh = construction_Zred_Fred_dZreddw_dFreddw(omega_idx,M,C,K,F,Nh,beta,gamma,ddl_ln,ddl_nl,derivatives,penalite,nu)
    fnl_tilde = Fh - Z@Xhs_RL_idx

    """
    Calcul de dfnltilde_dxtilde => Renvoie un truc qui a l'air bon
    """
    Xhs_RL_idx_torch = torch.tensor(Xhs_RL_idx,requires_grad=True, dtype=torch.float32).unsqueeze(0)
    fnl_tilde_torch = fnl_tilde_rlhbm(Xhs_RL_idx_torch,IDFT_1ddl_tensor,DFT_1ddl_tensor,g0,kn,eps, Nt, device='cpu')
    fnl_tilde_torch_sum = fnl_tilde_torch.sum()
    fnl_tilde_torch_sum.backward()
    print(Xhs_RL_idx_torch.grad)

    """
    Affichage des signaux temporels déplacement et effort => OK
    """
    ax_dep.plot(t,IDFT_1ddl@Xhs_RL_idx,color=colors_poly['orange_poly'])
    ax_dep.plot(t,[g0]*Nt,'r--')
    ax_dep.fill_between(t,[g0]*Nt,[0.8]*Nt,color='gray',alpha=0.4)
    ax_dep.set_ylim(top=0.7)
    ax_dep.set_xlim(0,2*np.pi/omega_idx*(1-1/Nt))
    ax_dep.set_ylabel('déplacement')
    # effort issu de la résolution classique
    ax_eff.plot(t,IDFT_1ddl@fnl_tilde,color=colors_poly['orange_poly'])
    # effort calculé par la fonction pytorch pour le déplacement Xhs_RL_idx
    ax_eff.plot(t,torch.mv(IDFT_1ddl_tensor,fnl_tilde_torch.squeeze()).detach().numpy(),'--',color=colors_poly['red_poly'])

    ax_eff.set_ylabel('effort')
    ax_eff.set_xlim(0,2*np.pi/omega_idx*(1-1/Nt))

    """
    Affichage de la courbe de réponse linéaire en noire et non-linéaire en orange
    le point rouge correspond aux signaux tracés sur l'autre figure
    """
    plt.figure()
    plt.plot(omegas,amps_lin,'k')
    plt.plot(omegas,[g0]*n_pts,'r--')
    plt.plot(omegas_RL,amps_RL[:,0],color=colors_poly['orange_poly'])
    plt.plot(omegas_RL[idx_RL],amps_RL[idx_RL,0],'o',color=colors_poly['red_poly'])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$||x_2(t)||_\infty$')

    plt.show()
