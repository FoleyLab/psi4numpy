"""
Simple demonstration of CQED-RHF method on the water molecule
coupled to a strong photon field with comparison to results from 
code in the hilbert package described in [DePrince:2021:094112] and available
at https://github.com/edeprince3/hilbert

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, and helper_CQED_RHF <==
import psi4
import numpy as np
from helper_CQED_KS import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2

mol = psi4.geometry("""
He 0 0 -5
He 0 0  5
symmetry c1
""")
options = {'BASIS':               'aug-cc-pVDZ',
           'DFT_SPHERICAL_POINTS': 110,
           'DFT_RADIAL_POINTS':    20}

coef_C = 0.0093
coef_B = 5.9
coef_beta = 1.0 / 32.0 * (3.0 / (coef_B ** 2.0)) ** (3.0 / 4.0)

def compute_vv10_kernel(rho, gamma):
    kappa_pref = coef_B * (1.5 * np.pi) / ((9.0 * np.pi) ** (1.0 / 6.0))
    
    # Compute R quantities
    Wp = (4.0 / 3.0) * np.pi * rho
    Wg = coef_C * ((gamma / (rho * rho)) ** 2.0)
    W0 = np.sqrt(Wg + Wp)
    
    kappa = rho ** (1.0 / 6.0) * kappa_pref
    return W0, kappa

def compute_vv10(D, Vpot):


    nbf = D.shape[0]
    Varr = np.zeros((nbf, nbf))
    
    total_e = 0.0
    tD = 2.0 * np.array(D)
    
    points_func = Vpot.properties()[0]
    superfunc = Vpot.functional()

    xc_e = 0.0
    vv10_e = 0.0
    
    # First loop over the outer set of blocks
    for l_block in range(Vpot.nblocks()):
        
        # Obtain general grid information
        l_grid = Vpot.get_block(l_block)
        l_w = np.array(l_grid.w())
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())
        l_npoints = l_w.shape[0]

        points_func.compute_points(l_grid)

        
        # Compute the functional itself
        ret = superfunc.compute_functional(points_func.point_values(), -1)
        
        xc_e += np.vdot(l_w, np.array(ret["V"])[:l_npoints])
        v_rho = np.array(ret["V_RHO_A"])[:l_npoints]
        v_gamma = np.array(ret["V_GAMMA_AA"])[:l_npoints]
        
        # Begin VV10 information
        l_rho = np.array(points_func.point_values()["RHO_A"])[:l_npoints]
        l_gamma = np.array(points_func.point_values()["GAMMA_AA"])[:l_npoints]
        
        l_W0, l_kappa = compute_vv10_kernel(l_rho, l_gamma)
        
        phi_kernel = np.zeros_like(l_rho)
        phi_U = np.zeros_like(l_rho)
        phi_W = np.zeros_like(l_rho)
        
        # Loop over the inner set of blocks
        for r_block in range(Vpot.nblocks()):
            
            # Repeat as for the left blocks
            r_grid = Vpot.get_block(r_block)
            r_w = np.array(r_grid.w())
            r_x = np.array(r_grid.x())
            r_y = np.array(r_grid.y())
            r_z = np.array(r_grid.z())
            r_npoints = r_w.shape[0]

            points_func.compute_points(r_grid)

            r_rho = np.array(points_func.point_values()["RHO_A"])[:r_npoints]
            r_gamma = np.array(points_func.point_values()["GAMMA_AA"])[:r_npoints]
        
            r_W0, r_kappa = compute_vv10_kernel(r_rho, r_gamma)
            
            # Build the distnace matrix
            R2  = (l_x[:, None] - r_x) ** 2
            R2 += (l_y[:, None] - r_y) ** 2
            R2 += (l_z[:, None] - r_z) ** 2
            
            # Build g
            g = l_W0[:, None] * R2 + l_kappa[:, None]
            gp = r_W0 * R2 + r_kappa
        
            # 
            F_kernal = -1.5 * r_w * r_rho / (g * gp * (g + gp))
            F_U = F_kernal * ((1.0 / g) + (1.0 / (g + gp)))
            F_W = F_U * R2


            phi_kernel += np.sum(F_kernal, axis=1)
            phi_U += -np.sum(F_U, axis=1)
            phi_W += -np.sum(F_W, axis=1)
                        
        # Compute those derivatives
        kappa_dn = l_kappa / (6.0 * l_rho)
        w0_dgamma = coef_C * l_gamma / (l_W0 * l_rho ** 4.0)
        w0_drho = 2.0 / l_W0 * (np.pi/3.0 - coef_C * np.power(l_gamma, 2.0) / (l_rho ** 5.0))

        # Sum up the energy
        vv10_e += np.sum(l_w * l_rho * (coef_beta + 0.5 * phi_kernel))

        # Perturb the derivative quantities
        v_rho += coef_beta + phi_kernel + l_rho * (kappa_dn * phi_U + w0_drho * phi_W)
        v_rho *= 0.5
        
        v_gamma += l_rho * w0_dgamma * phi_W

        # Recompute to l_grid
        lpos = np.array(l_grid.functions_local_to_global())
        points_func.compute_points(l_grid)
        nfunctions = lpos.shape[0]
        
        # Integrate the LDA and GGA quantities
        phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :nfunctions]
        phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :nfunctions]
        phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :nfunctions]
        phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :nfunctions]
        
        # LDA
        Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho, l_w, phi, optimize=True)

                # GGA
        l_rho_x = np.array(points_func.point_values()["RHO_AX"])[:l_npoints]
        l_rho_y = np.array(points_func.point_values()["RHO_AY"])[:l_npoints]
        l_rho_z = np.array(points_func.point_values()["RHO_AZ"])[:l_npoints]
        
        tmp_grid = 2.0 * l_w * v_gamma
        Vtmp += np.einsum('pb,p,p,pa->ab', phi, tmp_grid, l_rho_x, phi_x, optimize=True)
        Vtmp += np.einsum('pb,p,p,pa->ab', phi, tmp_grid, l_rho_y, phi_y, optimize=True)
        Vtmp += np.einsum('pb,p,p,pa->ab', phi, tmp_grid, l_rho_z, phi_z, optimize=True)
        
        # Sum back to the correct place
        Varr[(lpos[:, None], lpos)] += Vtmp + Vtmp.T
        
    print("   VV10 NL energy: %16.8f" % vv10_e)
        
    xc_e += vv10_e
    return xc_e, Varr

lam = np.array([0.0, 0.00, 0.00])
ks_solver(lam, "VV10", mol, options, compute_vv10, jk_type="PK")       
