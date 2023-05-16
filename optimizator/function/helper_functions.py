import torch
import numpy as np
from scipy.linalg import eigh

def amort_modal(K,M,xi,seuil_xi=np.array([]),fact_xi=np.array([])):
    """
    Construction de la matrice d'amortissement modal

    Parameters
    ----------
    K : 2d-array of float64 : matrice de raideur
    M : 2d-array of float64 : matrice de masse
    xi : (1) float64 : coefficient d'amortissement des seuil_xi[0] premiers modes,
                                   **si** spécification de seuil_xi et fact_xi non-vides
         (2) float64 : amortissement modal uniforme **si** de seuil_xi et fact_xi vides
         (3) 1d-array of float64 : coefficients d'amortissement de l'intégralité des M.shape[1] modes
    seuil_xi : 1d-array : seuils à partir desquels les coefficients xi sont multipliés par fact_xi
    fact_xi  : 1d-array : cf. seuil_xi. len(seuil_xi) == len(fact_xi)

    Returns
    -------
    C : 2d-array of float64 : matrice d'amortissement modal

    """

    # Forçage de la symetrie pour meilleure precision numerique
    K = (K + K.T)/2 ; M = (M + M.T)/2

    # résolution du problème aux valeurs propres
    wi_carre,phi = eigh(K,M)

    # M-orthonormalisation des modes propres
    M_norm = np.einsum('ji,jk,kl->i',phi,M,phi,optimize=True) # phi.T M phi
    phi = phi/np.sqrt(M_norm)

    wi = np.sqrt(wi_carre.real)
    idx = np.argsort(wi)
    # réorganisation par valeurs croissantes
    wi = wi[idx] ; phi = phi[:,idx]

    # (1)
    if seuil_xi.size and fact_xi.size :
        print('> Amortissement modal non-uniforme',flush=True)
        Cm, xi = mult_seuil_fact_xi(wi,xi,seuil_xi,fact_xi)

    # (2) et (3)
    else:
        if isinstance(xi,np.ndarray):
            print('> Amortissement modal intégralement spécifié',flush=True)
        elif isinstance(xi,float):
            print('> Amortissement modal uniforme',flush=True)
        Cm = np.diag(2*xi*wi)

    print('> ξ_i =',xi,flush=True)

    C = np.linalg.inv(phi.T) @ Cm @ np.linalg.inv(phi)
    # Forçage de la symetrie pour meilleure precision numerique
    C = (C + C.T)/2

    return C

# @jit(nopython=True, fastmath=True)
def mult_seuil_fact_xi(wi,xi,seuil_xi,fact_xi):
    xi_range = np.full(len(wi),xi)
    for i,seuil in enumerate(seuil_xi):
        xi_range[seuil_xi[i]:] = xi*fact_xi[i]
    # print(xi_range)
    return np.diag(2*xi_range*wi),xi_range

def construction_IDFT(Nt,Nh,nb_ddl):
	IDFT_1ddl = np.zeros((Nt,2*Nh + 1),dtype=np.float64)
	IDFT_1ddl[:,0] = 1/2
	for i in range(1,Nh + 1):
		t_i = np.linspace(0,i*2*np.pi*(1- 1/Nt),Nt)
		IDFT_1ddl[:,2*i - 1] = np.cos(t_i)
		IDFT_1ddl[:,2*i] = np.sin(t_i)
	return np.kron(IDFT_1ddl,np.eye(nb_ddl,dtype=np.float64))

def construction_DFT(Nt,Nh,nb_ddl):
	IDFT_1ddl = construction_IDFT(Nt,Nh,1)
	IDFT_1ddl[:,0] *= 2
	return 2/Nt*np.kron(np.transpose(IDFT_1ddl),np.eye(nb_ddl,dtype=np.float64))

# @jit(nopython=True, fastmath=True)
def construction_Zred_Fred_dZreddw_dFreddw(omega,M,C,K,F,Nh,beta,gamma,ddl_ln,ddl_nl,derivatives=False,penalite=None,nu=1):
	# Taille du système
	n_dof = len(M)
	n_dof_ln = len(ddl_ln)
	n_dof_nl = len(ddl_nl)
	HBM_system_size_red = n_dof_nl*(2*Nh+1)
	HBM_system_size_full = n_dof*(2*Nh+1)
	# Allocation des matrices
	Z_red = np.zeros((HBM_system_size_red,HBM_system_size_red),dtype=np.float64)
	F_red = np.zeros(HBM_system_size_red,dtype=np.float64)
	if penalite != None:
		Zr_k = np.zeros((2*n_dof_nl,2*n_dof_nl),dtype=np.float64)
		Zr = np.zeros((HBM_system_size_red,HBM_system_size_red),dtype=np.float64)
		Fr = np.zeros(HBM_system_size_red,dtype=np.float64)

	if derivatives == True:
		dZ_reddw = np.zeros((HBM_system_size_red,HBM_system_size_red),dtype=np.float64)
		dF_reddw = np.zeros(HBM_system_size_red,dtype=np.float64)
		if penalite!=None:
			dZr_dw = np.zeros((HBM_system_size_red,HBM_system_size_red),dtype=np.float64)
			dFr_dw = np.zeros(HBM_system_size_red,dtype=np.float64)
	Z = np.zeros((HBM_system_size_full,HBM_system_size_full),dtype=np.float64)
	Fh = np.zeros(HBM_system_size_full,dtype=np.float64)

	Z_k = np.zeros((2*n_dof,2*n_dof),dtype=np.float64)
	Fh_k = np.zeros(2*n_dof,dtype=np.float64)

	Z[:n_dof,:n_dof] = beta*K
	if n_dof_ln > 0:
		Z_k_ln = np.zeros((2*n_dof_ln,2*n_dof_ln),dtype=np.float64)
		Z_k_ln_nl = np.zeros((2*n_dof_ln,2*n_dof_nl),dtype=np.float64)
		Z_k_nl = np.zeros((2*n_dof_nl,2*n_dof_nl),dtype=np.float64)
		Z_k_nl_ln = np.zeros((2*n_dof_nl,2*n_dof_ln),dtype=np.float64)
		Fh_k_nl = np.zeros(2*n_dof_nl,dtype=np.float64)
		Fh_k_ln = np.zeros(2*n_dof_ln,dtype=np.float64)


		dZdw_k_ln = np.zeros((2*n_dof_ln,2*n_dof_ln),dtype=np.float64)
		dZdw_k_ln_nl = np.zeros((2*n_dof_ln,2*n_dof_nl),dtype=np.float64)
		dZdw_k_nl = np.zeros((2*n_dof_nl,2*n_dof_nl),dtype=np.float64)
		dZdw_k_nl_ln = np.zeros((2*n_dof_nl,2*n_dof_ln),dtype=np.float64)

		Z_red_k = np.zeros((2*n_dof_nl,2*n_dof_nl),dtype=np.float64)

		M_ln, C_ln, K_ln = M[ddl_ln,:][:,ddl_ln], C[ddl_ln,:][:,ddl_ln], K[ddl_ln,:][:,ddl_ln]
		M_nl, C_nl, K_nl = M[ddl_nl,:][:,ddl_nl], C[ddl_nl,:][:,ddl_nl], K[ddl_nl,:][:,ddl_nl]
		M_nl_ln, C_nl_ln, K_nl_ln = M[ddl_nl,:][:,ddl_ln], C[ddl_nl,:][:,ddl_ln], K[ddl_nl,:][:,ddl_ln]
		M_ln_nl, C_ln_nl, K_ln_nl = M[ddl_ln,:][:,ddl_nl], C[ddl_ln,:][:,ddl_nl], K[ddl_ln,:][:,ddl_nl]

		F_ln = F[ddl_ln]
		F_nl = F[ddl_nl]
		# Remplissage de la matrice selon son expression analytique (cf JOANNIN C.)
		K_inv_ln_K_ln_nl = np.ascontiguousarray(np.linalg.solve(beta*K_ln,beta*K_ln_nl))
		Z_red[:n_dof_nl,:n_dof_nl] = beta*K_nl - np.dot(beta*K_nl_ln,K_inv_ln_K_ln_nl)
	else:
		dZdw_k = np.zeros((2*n_dof_nl,2*n_dof_nl),dtype=np.float64)
		Z_red[:n_dof,:n_dof] = Z[:n_dof,:n_dof]

	if penalite != None:
		Z_pen_0_inv = np.zeros((1,1))
		Z_pen_0_inv[0,0] = 1/(beta*penalite)
		Zr[:n_dof_nl,:n_dof_nl] = np.linalg.inv(np.linalg.inv(Z_red[:n_dof_nl,:n_dof_nl]) + Z_pen_0_inv )

	for k in range(1,Nh+1):
		#Construction du bloc lié à l'harmonique K Z
		Z_k[:n_dof,:n_dof] = beta*K - (beta/gamma**2)*((k*omega)**2)*M
		Z_k[n_dof:,n_dof:] = Z_k[:n_dof,:n_dof]
		Z_k[n_dof:,:n_dof] = -(beta/gamma)*k*omega*C
		Z_k[:n_dof,n_dof:] = -Z_k[n_dof:,:n_dof]

		Z[k*2*n_dof-n_dof:(k+1)*2*n_dof-n_dof,k*2*n_dof-n_dof:(k+1)*2*n_dof-n_dof] = Z_k
		if n_dof_ln > 0:
			#Construction du bloc lié à l'harmonique K Z_k_ln
			Z_k_ln[:n_dof_ln,:n_dof_ln] = beta*K_ln - (beta/gamma**2)*((k*omega)**2)*M_ln
			Z_k_ln[n_dof_ln:,n_dof_ln:] = Z_k_ln[:n_dof_ln,:n_dof_ln]
			Z_k_ln[n_dof_ln:,:n_dof_ln] = -(beta/gamma)*k*omega*C_ln
			Z_k_ln[:n_dof_ln,n_dof_ln:] = -Z_k_ln[n_dof_ln:,:n_dof_ln]
			#Construction du bloc lié à l'harmonique K Z_k_ln_nl
			Z_k_ln_nl[:n_dof_ln,:n_dof_nl] = beta*K_ln_nl - (beta/gamma**2)*((k*omega)**2)*M_ln_nl
			Z_k_ln_nl[n_dof_ln:,n_dof_nl:] = Z_k_ln_nl[:n_dof_ln,:n_dof_nl]
			Z_k_ln_nl[n_dof_ln:,:n_dof_nl] = -(beta/gamma)*k*omega*C_ln_nl
			Z_k_ln_nl[:n_dof_ln,n_dof_nl:] = -Z_k_ln_nl[n_dof_ln:,:n_dof_nl]

			#Construction du bloc lié à l'harmonique K Z_k_nl
			Z_k_nl[:n_dof_nl,:n_dof_nl] = beta*K_nl - (beta/gamma**2)*((k*omega)**2)*M_nl
			Z_k_nl[n_dof_nl:,n_dof_nl:] = Z_k_nl[:n_dof_nl,:n_dof_nl]
			Z_k_nl[n_dof_nl:,:n_dof_nl] = -(beta/gamma)*k*omega*C_nl
			Z_k_nl[:n_dof_nl,n_dof_nl:] = -Z_k_nl[n_dof_nl:,:n_dof_nl]

			#Construction du bloc lié à l'harmonique K Z_k_nl_ln
			Z_k_nl_ln[:n_dof_nl,:n_dof_ln] = beta*K_nl_ln - (beta/gamma**2)*((k*omega)**2)*M_nl_ln
			Z_k_nl_ln[n_dof_nl:,n_dof_ln:] = Z_k_nl_ln[:n_dof_nl,:n_dof_ln]
			Z_k_nl_ln[n_dof_nl:,:n_dof_ln] = -(beta/gamma)*k*omega*C_nl_ln
			Z_k_nl_ln[:n_dof_nl,n_dof_ln:] = -Z_k_nl_ln[n_dof_nl:,:n_dof_ln]


		# Assemblage du bloc lié à l'harmonique K
		if n_dof_ln > 0:
			Z_k_inv_ln_Z_k_ln_nl = np.ascontiguousarray(np.linalg.solve(Z_k_ln,Z_k_ln_nl))
			Z_red_k = Z_k_nl - np.dot(Z_k_nl_ln,Z_k_inv_ln_Z_k_ln_nl)
			Z_red[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl,k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = Z_red_k
		else:
			Z_red[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl,k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = Z_k

		if penalite != None:
			if n_dof_ln > 0:
				# print('====\n',omega,'\n',Z_red_k,'\n')
				Z_red_k_inv = np.linalg.inv(Z_red_k)
			else:
				Z_red_k_inv = np.linalg.inv(Z_k)

			Z_pen_k = np.zeros((2,2))
			Z_pen_k[0,0] = beta*penalite
			Z_pen_k[1,1] = beta*penalite
			Z_pen_k_inv = np.linalg.inv(Z_pen_k)

			Zr_k = np.linalg.inv(Z_red_k_inv + Z_pen_k_inv)
			Zr[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl,k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = Zr_k
		if k == nu and nu>=1:
			Fh_k[:n_dof] = F
			Fh[k*2*n_dof-n_dof:(k+1)*2*n_dof-n_dof] = Fh_k
			if n_dof_ln > 0:
				Fh_k_ln[:n_dof_ln] = F_ln
				Fh_k_nl[:n_dof_nl] = F_nl
				Z_k_inv_ln_Fh_k_ln = np.linalg.solve(Z_k_ln,Fh_k_ln)
				F_k_red = Fh_k_nl - np.dot(Z_k_nl_ln,Z_k_inv_ln_Fh_k_ln)
				F_red[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = F_k_red
			else:
				F_red[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = Fh_k
				F_k_red = Fh_k
			if penalite != None:
				Fr_k = Zr_k@Z_red_k_inv@F_k_red
				Fr[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = Fr_k
		if derivatives == True:
			if n_dof_ln > 0:
				dZdw_k_nl[:n_dof_nl,:n_dof_nl] = -2*(beta/gamma**2)*(omega*(k**2))*M_nl
				dZdw_k_nl[n_dof_nl:,n_dof_nl:] = dZdw_k_nl[:n_dof_nl,:n_dof_nl]
				dZdw_k_nl[n_dof_nl:,:n_dof_nl] = -(beta/gamma)*k*C_nl
				dZdw_k_nl[:n_dof_nl,n_dof_nl:] = -dZdw_k_nl[n_dof_nl:,:n_dof_nl]

				dZdw_k_ln_nl[:n_dof_ln,:n_dof_nl] = -2*(beta/gamma**2)*(omega*(k**2))*M_ln_nl
				dZdw_k_ln_nl[n_dof_ln:,n_dof_nl:] = dZdw_k_ln_nl[:n_dof_ln,:n_dof_nl]
				dZdw_k_ln_nl[n_dof_ln:,:n_dof_nl] = -(beta/gamma)*k*C_ln_nl
				dZdw_k_ln_nl[:n_dof_ln,n_dof_nl:] = -dZdw_k_ln_nl[n_dof_ln:,:n_dof_nl]

				dZdw_k_ln[:n_dof_ln,:n_dof_ln] = -2*(beta/gamma**2)*(omega*(k**2))*M_ln
				dZdw_k_ln[n_dof_ln:,n_dof_ln:] = dZdw_k_ln[:n_dof_ln,:n_dof_ln]
				dZdw_k_ln[n_dof_ln:,:n_dof_ln] = -(beta/gamma)*k*C_ln
				dZdw_k_ln[:n_dof_ln,n_dof_ln:] = -dZdw_k_ln[n_dof_ln:,:n_dof_ln]

				dZdw_k_nl_ln[:n_dof_nl,:n_dof_ln] = -2*(beta/gamma**2)*(omega*(k**2))*M_nl_ln
				dZdw_k_nl_ln[n_dof_nl:,n_dof_ln:] = dZdw_k_nl_ln[:n_dof_nl,:n_dof_ln]
				dZdw_k_nl_ln[n_dof_nl:,:n_dof_ln] = -(beta/gamma)*k*C_nl_ln
				dZdw_k_nl_ln[:n_dof_nl,n_dof_ln:] = -dZdw_k_nl_ln[n_dof_nl:,:n_dof_ln]

				terme1 = -np.dot(dZdw_k_nl_ln,Z_k_inv_ln_Z_k_ln_nl)
				Z_k_ln_inv = np.linalg.inv(Z_k_ln)
				dZ_k_ln_inv_dw = -np.dot(Z_k_ln_inv,np.dot(dZdw_k_ln,Z_k_ln_inv))
				terme2 = -np.dot(Z_k_nl_ln,np.dot(dZ_k_ln_inv_dw,Z_k_ln_nl))
				terme3 = -np.dot(Z_k_nl_ln,np.dot(Z_k_ln_inv,dZdw_k_ln_nl))

				dZ_k_reddw = dZdw_k_nl + terme1 + terme2 + terme3
				dZ_reddw[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl,k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = dZ_k_reddw
			else:
				dZdw_k[:n_dof_nl,:n_dof_nl] = -2*(beta/gamma**2)*(omega*(k**2))*M
				dZdw_k[n_dof_nl:,n_dof_nl:] = dZdw_k[:n_dof_nl,:n_dof_nl]
				dZdw_k[n_dof_nl:,:n_dof_nl] = -(beta/gamma)*k*C
				dZdw_k[:n_dof_nl,n_dof_nl:] = -dZdw_k[n_dof_nl:,:n_dof_nl]
				dZ_k_reddw = dZdw_k
			if penalite != None:
				dZr_k_dw = Zr_k@Z_red_k_inv@dZ_k_reddw@Z_red_k_inv@Zr_k
				dZr_dw[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl,k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl] = dZr_k_dw

			if k == nu and nu>=1:
				if n_dof_ln > 0 :
					dF_k_reddw = -np.dot(np.dot(dZdw_k_nl_ln,Z_k_ln_inv) + np.dot(Z_k_nl_ln,dZ_k_ln_inv_dw),Fh_k_ln)
					dF_reddw[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl]  = dF_k_reddw
				if penalite != None:
					dF_k_reddw = np.zeros(Z_red_k_inv.shape[0])
					dFr_k_dw = dZr_k_dw@Z_red_k_inv@F_k_red + Zr_k@(Z_red_k_inv@dF_k_reddw - Z_red_k_inv@dZ_k_reddw@Z_red_k_inv@F_k_red)
					dFr_dw[k*2*n_dof_nl-n_dof_nl:(k+1)*2*n_dof_nl-n_dof_nl]  = dFr_k_dw

	return Z_red,F_red

def fnl_tilde_rlhbm(Xh_red,IDFT_1ddl,DFT_1ddl,g0,kn,eps,Nt):
	g = torch.mm(IDFT_1ddl,Xh_red.t())  - g0
	fnl = kn/2*g + torch.sqrt((kn*g/2)**2 + eps**2)
	fnl_tilde = torch.mm(DFT_1ddl,fnl).t()
	return fnl_tilde