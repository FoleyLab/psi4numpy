#! A simple Psi4 input script to compute a SCF reference using Psi4's libJK

import time
import numpy as np
import psi4

from pkg_resources import parse_version
if parse_version(psi4.__version__) >= parse_version('1.3a1'):
    build_superfunctional = psi4.driver.dft.build_superfunctional
else:
    build_superfunctional = psi4.driver.dft_funcs.build_superfunctional

# Diagonalize routine
def build_orbitals(diag, A, ndocc):
    Fp = psi4.core.triplet(A, diag, A, True, False, True)

    nbf = A.shape[0]
    Cp = psi4.core.Matrix(nbf, nbf)
    eigvecs = psi4.core.Vector(nbf)
    Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.doublet(A, Cp, False, False)

    Cocc = psi4.core.Matrix(nbf, ndocc)
    Cocc.np[:] = C.np[:, :ndocc]

    D = psi4.core.doublet(Cocc, Cocc, False, True)
    return C, Cocc, D, eigvecs

def ks_solver(lambda_vector, alias, mol, options, V_builder, jk_type="DF", output="output.dat", restricted=True):

    # Build our molecule
    mol = mol.clone()
    mol.reset_point_group('c1')
    mol.fix_orientation(True)
    mol.fix_com(True)
    mol.update_geometry()

    # Set options
    psi4.set_output_file(output)

    psi4.core.prepare_options_for_module("SCF")
    psi4.set_options(options)
    psi4.core.set_global_option("SCF_TYPE", jk_type)

    maxiter = 20
    E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE") 
    D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")
    
    # Integral generation from Psi4's MintsHelper
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    mints = psi4.core.MintsHelper(wfn.basisset())
    S = mints.ao_overlap()

    # Build the V Potential
    sup = build_superfunctional(alias, restricted)[0]
    sup.set_deriv(2)
    sup.allocate()
    
    vname = "RV"
    if not restricted:
        vname = "UV"
    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, vname)
    Vpot.initialize()
    
    # Get nbf and ndocc for closed shell molecules
    nbf = wfn.nso()
    ndocc = wfn.nalpha()
    if wfn.nalpha() != wfn.nbeta():
        raise PsiException("Only valid for RHF wavefunctions!")
    
    print('\nNumber of occupied orbitals: %d' % ndocc)
    print('Number of basis functions:   %d' % nbf)
    
    # Build H_core for molecule only
    V = mints.ao_potential()
    T = mints.ao_kinetic()
    H = T.clone()
    H.add(V)

    
    # Orthogonalizer A = S^(-1/2)
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    
    # Build core orbitals
    C, Cocc, D, eigs = build_orbitals(H, A, ndocc)

    D_np = np.asarray(D)

    # Extra terms for Pauli-Fierz Hamiltonian
    # nuclear dipole
    mu_nuc = np.array([mol.nuclear_dipole()[0], mol.nuclear_dipole()[1], mol.nuclear_dipole()[2]])

    # electronic dipole integrals in AO basis
    mu_ao_x = mints.ao_dipole()[0]
    mu_ao_y = mints.ao_dipole()[1]
    mu_ao_z = mints.ao_dipole()[2]

    l_dot_mu_el = S.clone()
    l_dot_mu_el.axpy(-1, S)
    l_dot_mu_el.axpy(lambda_vector[0], mu_ao_x)
    l_dot_mu_el.axpy(lambda_vector[1], mu_ao_y)
    l_dot_mu_el.axpy(lambda_vector[2], mu_ao_z)

    # canonincal RHF density
    mu_exp = np.array([np.einsum("pq,pq->", 2 * np.asarray(mu_ao_x), D_np),
    np.einsum("pq,pq->", 2 * np.asarray(mu_ao_y), D_np), 
    np.einsum("pq,pq->", 2 * np.asarray(mu_ao_z), D_np) ])

    # need to add the nuclear term to the sum over the electronic dipole integrals
    mu_exp += mu_nuc
    rhf_dipole_moment = np.copy(mu_exp)


    # We need to carry around the electric field dotted into the nuclear dipole moment
    # and the electric field dotted into the RHF electronic dipole expectation value
    # see prefactor to sum of Line 3 of Eq. (9) in [McTague:2021:ChemRxiv]

    # \lambda_vector \cdot \mu_{nuc}
    l_dot_mu_nuc = np.dot(lambda_vector, mu_nuc)

    # \lambda_vecto \cdot < \mu > where <\mu> contains electronic and nuclear contributions
    l_dot_mu_exp = np.dot(lambda_vector, mu_exp)

    # dipole energy, Eq. (14) in [McTague:2021:ChemRxiv]
    #  0.5 * (\lambda_vector \cdot \mu_{nuc})** 2
    #      - (\lambda_vector \cdot <\mu> ) ( \lambda_vector\cdot \mu_{nuc})
    # +0.5 * (\lambda_vector \cdot <\mu>) ** 2
    d_c = (
        0.5 * l_dot_mu_nuc ** 2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp ** 2
    )

    # quadrupole arrays
    Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
    Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
    Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
    Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
    Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
    Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

    # Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
    Q_PF = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
    Q_PF -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy
    Q_PF -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

    # accounting for the fact that Q_ij = Q_ji
    # by weighting Q_ij x 2 which cancels factor of 1/2
    Q_PF -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
    Q_PF -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
    Q_PF -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

    Q_PF = psi4.core.Matrix.from_array(Q_PF)

    # Pauli-Fierz 1-e dipole terms scaled by
    # (\lambda_vector \cdot \mu_{nuc} - \lambda_vector \cdot <\mu>)
    # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
    d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el
    d_PF_p4 = psi4.core.Matrix.from_array(d_PF)
    
    # Setup data for DIIS
    t = time.time()
    E = 0.0
    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0
    
    # Initialize the JK object
    jk = psi4.core.JK.build(wfn.basisset())
    jk.set_memory(int(1.25e8))  # 1GB
    jk.initialize()
    jk.print_header()
    
    diis_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")
    
    print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))
    
    print('\nStarting SCF iterations:')
    t = time.time()
   
    print("\n    Iter            Energy             XC E         Delta E        D RMS\n")
    for SCF_ITER in range(1, maxiter + 1):
    
        # Compute JK
        jk.C_left_add(Cocc)
        jk.compute()
        jk.C_clear()
    
        # Build Fock matrix
        F = H.clone()
        F.axpy(2.0, jk.J()[0])
        F.axpy(-Vpot.functional().x_alpha(), jk.K()[0])

        # Build V
        ks_e = 0.0

        Vpot.set_D([D])
        Vpot.properties()[0].set_pointers(D)
        V = V_builder(D, Vpot)
        if V is None:
            ks_e = 0.0
        else:
            ks_e, V = V
            V = psi4.core.Matrix.from_array(V)
    
        F.axpy(1.0, V)

        D_np = np.asarray(D)
        # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
        M = psi4.core.Matrix.from_array(np.einsum("pq,rs,rs->pq", l_dot_mu_el, l_dot_mu_el, D_np))
        N = psi4.core.Matrix.from_array(np.einsum("pr,qs,rs->pq", l_dot_mu_el, l_dot_mu_el, D_np))
        
        F.add(Q_PF)
        F.add(d_PF_p4)
        F.axpy(2.0, M)
        F.axpy(-1.0, N)


        # DIIS error build and update
        diis_e = psi4.core.triplet(F, D, S, False, False, False)
        diis_e.subtract(psi4.core.triplet(S, D, F, False, False, False))
        diis_e = psi4.core.triplet(A, diis_e, A, False, False, False)
    
        diis_obj.add(F, diis_e)
    
        dRMS = diis_e.rms()

        # SCF energy and update
        SCF_E  = 2.0 * H.vector_dot(D)
        SCF_E += 2.0 * jk.J()[0].vector_dot(D)
        SCF_E -= Vpot.functional().x_alpha() * jk.K()[0].vector_dot(D)
        SCF_E += ks_e
        SCF_E += Enuc
    
        print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'
              % (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))
        if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
            break
    
        Eold = SCF_E
    
        # DIIS extrapolate
        F = diis_obj.extrapolate()
    
        # Diagonalize Fock matrix
        C, Cocc, D, eigs = build_orbitals(F, A, ndocc)

        # update electronic dipole expectation value
        mu_exp = np.array([np.einsum("pq,pq->", 2 * np.asarray(mu_ao_x), np.asarray(D)), 
        np.einsum("pq,pq->", 2 * np.asarray(mu_ao_y), np.asarray(D)),
        np.einsum("pq,pq->", 2 * np.asarray(mu_ao_z), np.asarray(D))])

        # add nuclear contribution 
        mu_exp += mu_nuc

        # update \lambda \cdot <\mu>
        l_dot_mu_exp = np.dot(lambda_vector, mu_exp)

        # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
        d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el
        d_PF_p4 = psi4.core.Matrix.from_array(d_PF)
    
        if SCF_ITER == maxiter:
            raise Exception("Maximum number of SCF cycles exceeded.")
    
    print('\nTotal time for SCF iterations: %.3f seconds ' % (time.time() - t))
    
    print('\nFinal SCF energy: %.8f hartree' % SCF_E)

    data = {}
    data["Da"] = D
    data["Ca"] = C
    data["eigenvalues"] = eigs
    return(SCF_E, data)
