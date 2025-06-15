import psi4
import numpy as np
import time
import json
import opt_einsum as oe

class CQEDRHFCalculator:
    def __init__(self, lambda_vector, molecule_string, psi4_options):
        self.lambda_vector = np.array(lambda_vector)
        self.molecule_string = molecule_string
        self.psi4_options = psi4_options

        # initialize results to None
        self.rhf_energy = None
        self.cqed_rhf_energy = None
        self.coefficients = None
        self.density_matrix = None
        self.fock_matrix = None
        self.orbital_energies = None
        self.rhf_dipole_moment = None
        self.dipole_moment = None
        self.nuclear_dipole_moment = None
        self.dipole_energy = None
        self.nuclear_repulsion_energy = None
        self.quadrupole_moment = None
        self.wfn = None
        self.qed_wfn = None

    def calc_cqed_rhf_energy(self):
        # define molecule and options
        mol = psi4.geometry(self.molecule_string)
        psi4.set_options(self.psi4_options)
        self.rhf_energy, wfn = psi4.energy("scf", return_wfn=True)
        self.psi4_wfn = wfn

        mints = psi4.core.MintsHelper(wfn.basisset())
        ndocc = wfn.nalpha()
        C = np.asarray(wfn.Ca())
        Cocc = C[:, :ndocc]
        D = oe.contract("pi,qi->pq", Cocc, Cocc, optimize="optimal")

        self.ndocc = ndocc
        self.n_orbitals = wfn.nmo()
        self.num_atoms = mol.natom()

        V = np.asarray(mints.ao_potential())
        T = np.asarray(mints.ao_kinetic())
        I = np.asarray(mints.ao_eri())

        mu_nuc_x = mol.nuclear_dipole()[0]
        mu_nuc_y = mol.nuclear_dipole()[1]
        mu_nuc_z = mol.nuclear_dipole()[2]

        # electronic dipole integrals in AO basis
        mu_ao_x = np.asarray(mints.ao_dipole()[0])
        mu_ao_y = np.asarray(mints.ao_dipole()[1])
        mu_ao_z = np.asarray(mints.ao_dipole()[2])

        #mu_nuc_x, mu_nuc_y, mu_nuc_z = mol.nuclear_dipole()
        mu_nuc = np.array([mu_nuc_x, mu_nuc_y, mu_nuc_z])

        #mu_ao_x, mu_ao_y, mu_ao_z = [np.asarray(x) for x in mints.ao_dipole()]
        mu_ao = np.array([mu_ao_x, mu_ao_y, mu_ao_z])

        d_ao = sum(self.lambda_vector[i] * mu_ao[i] for i in range(3))

        mu_exp = np.array([
            2 * oe.contract("pq,pq->", mu_ao[i], D, optimize='optimal') for i in range(3)
        ]) + mu_nuc

        self.rhf_dipole_moment = mu_exp.copy()
        d_nuc = sum(self.lambda_vector[i] * mu_nuc[i] for i in range(3))
        d_exp = sum(self.lambda_vector[i] * mu_exp[i] for i in range(3))

        d_c = 0.5 * d_nuc**2 - d_nuc * d_exp + 0.5 * d_exp**2

        self.dipole_energy = d_c

        Q_ao_xx, Q_ao_xy, Q_ao_xz, Q_ao_yy, Q_ao_yz, Q_ao_zz = [np.asarray(x) for x in mints.ao_quadrupole()]
        Q_ao = [Q_ao_xx, Q_ao_xy, Q_ao_xz, Q_ao_yy, Q_ao_yz, Q_ao_zz]

        Q_PF = -0.5 * self.lambda_vector[0]**2 * Q_ao_xx
        Q_PF -= 0.5 * self.lambda_vector[1]**2 * Q_ao_yy
        Q_PF -= 0.5 * self.lambda_vector[2]**2 * Q_ao_zz
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[1] * Q_ao_xy
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[2] * Q_ao_xz
        Q_PF -= self.lambda_vector[1] * self.lambda_vector[2] * Q_ao_yz


        d_PF = (d_nuc - d_exp) * d_ao 
        H_0 = T + V
        H = H_0 + Q_PF 

        S = mints.ao_overlap()
        A = mints.ao_overlap()
        A.power(-0.5, 1.0e-16)
        A = np.asarray(A)

        Eold = 0.0
        Enuc = mol.nuclear_repulsion_energy()
        self.nuclear_repulsion_energy = Enuc

        maxiter = 500
        E_conv = self.psi4_options.get("e_convergence", 1.0e-7)
        D_conv = self.psi4_options.get("d_convergence", 1.0e-5)

        for scf_iter in range(1, maxiter + 1):
            J = oe.contract("pqrs,rs->pq", I, D, optimize="optimal")
            K = oe.contract("prqs,rs->pq", I, D, optimize="optimal")
            
            N = oe.contract("pr,qs,rs->pq", d_ao, d_ao, D, optimize="optimal")

            F = H + 2 * J - K - N 

            diis_e = A @ (F @ D @ S - S @ D @ F) @ A
            dRMS = np.sqrt(np.mean(diis_e**2))
            E_scf = oe.contract("pq,pq->", F + H, D, optimize="optimal") + Enuc 

            if abs(E_scf - Eold) < E_conv and dRMS < D_conv:
                break
            Eold = E_scf

            Fp = A @ F @ A
            e, C2 = np.linalg.eigh(Fp)
            C = A @ C2
            Cocc = C[:, :ndocc]
            D = np.einsum("pi,qi->pq", Cocc, Cocc)

            H = H_0 + Q_PF 

        else:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")
        
        # update the dipole expectation value with the converged density matrix
        mu_exp = np.array([
            2 * oe.contract("pq,pq->", mu_ao[i], D, optimize='optimal') for i in range(3)
        ]) + mu_nuc

        # update d_exp
        d_exp = sum(self.lambda_vector[i] * mu_exp[i] for i in range(3))
        d_c = 0.5 * d_nuc**2 - d_nuc * d_exp + 0.5 * d_exp**2
        self.dipole_energy = d_c
        d_PF = (d_nuc - d_exp) * d_ao

        # go ahead and grab the dipole and quadrupole moment gradients for later use
        c_origin = [0.0, 0.0, 0.0]
        max_order = 2
        Dp4 = psi4.core.Matrix.from_array(2 * D)
        
        self.quad_grad = np.asarray(mints.multipole_grad(Dp4, max_order, c_origin))


        self.cqed_rhf_energy = E_scf
        self.coefficients = C
        self.density_matrix = D
        self.fock_matrix = F
        self.orbital_energies = e
        self.dipole_moment = mu_exp
        self.nuclear_dipole_moment = mu_nuc
        self.Q_PF = Q_PF
        self.d_ao = d_ao
        self.d_PF = d_PF

        #q_exp = oe.contract([2 * np.einsum("pq,pq->", Q, D, optimize="optimal") for Q in Q_ao])
        #self.quadrupole_moment = q_exp

        #wfn_dict = psi4.core.Wavefunction.to_file(wfn)
        #wfn_dict['matrix']['Ca'] = np.copy(C)
        #wfn_dict['matrix']['Cb'] = np.copy(C)
        #wfn_dict['matrix']['Da'] = np.copy(D)
        #wfn_dict['matrix']['Db'] = np.copy(D)
        #wfn_dict['matrix']['Fa'] = np.copy(F)
        #wfn_dict['matrix']['Fb'] = np.copy(F)
        #wfn_dict['vector']['epsilon_a'] = np.copy(e)
        #wfn_dict['vector']['epsilon_b'] = np.copy(e)
        #self.qed_wfn = psi4.core.Wavefunction.from_file(wfn_dict)

    def summary(self):
        print(f"RHF Energy:           {self.rhf_energy:.8f} Ha")
        print(f"CQED-RHF Energy:      {self.cqed_rhf_energy:.8f} Ha")
        print(f"Dipole Energy:        {self.dipole_energy:.8f} Ha")
        print(f"Nuclear Repulsion:    {self.nuclear_repulsion_energy:.8f} Ha")
        print(f"Dipole Moment:        {self.dipole_moment}")

    def calc_scf_gradient(self, qed_wfn=False):
        """Calculate the SCF gradient using psi4 core functionality.
        
        This method requires that the wavefunction has been calculated first.
        One can use the default wfn from a psi4 calculation or can update with cqed-rhf quantities

        It returns the gradient as a numpy array.

        Raises:
            Exception: If the wavefunction has not been calculated.
        
        """
        if self.psi4_wfn is None:
            raise Exception("Wavefunction has not been calculated. Please run calc_cqed_rhf_energy() first.")

        if qed_wfn:
            # update the wavefunction with the CQED-RHF results
            self.psi4_wfn.Ca().nph[0][:,:] = psi4.core.Matrix.from_array(self.coefficients)
            self.psi4_wfn.Cb().nph[0][:,:] = psi4.core.Matrix.from_array(self.coefficients)
            self.psi4_wfn.Da().nph[0][:,:] = psi4.core.Matrix.from_array(self.density_matrix)
            self.psi4_wfn.Db().nph[0][:,:] = psi4.core.Matrix.from_array(self.density_matrix)
            self.psi4_wfn.epsilon_a().nph[0][:] = psi4.core.Vector.from_array(self.orbital_energies)
            self.psi4_wfn.epsilon_b().nph[0][:] = psi4.core.Vector.from_array(self.orbital_energies)

            self.scf_grad = np.asarray(psi4.core.scfgrad(self.psi4_wfn))
        else:
            self.scf_grad = np.asarray(psi4.core.scfgrad(self.psi4_wfn))

        
        return self.scf_grad
    
    def calc_quadrupole_gradient(self):
        """ Calculate the quadrupole gradient using the CQED-RHF results.
        Returns:
            numpy.ndarray: The quadrupole gradient as a numpy array.
        """
        self.o_dse_gradient = np.zeros(3 * self.num_atoms)

        for atom_index in range(self.num_atoms):
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[0] ** 2 * self.quad_grad[deriv_index, 3]
                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[1] ** 2 * self.quad_grad[deriv_index, 6]
                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[2] ** 2 * self.quad_grad[deriv_index, 8]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[0] * self.lambda_vector[1] * self.quad_grad[deriv_index, 4]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[0] * self.lambda_vector[2] * self.quad_grad[deriv_index, 5]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[1] * self.lambda_vector[2] * self.quad_grad[deriv_index, 7]

        print(self.o_dse_gradient)

        
    def export_to_json(self, filename):
        data = {
            "RHF Energy": self.rhf_energy,
            "CQED-RHF Energy": self.cqed_rhf_energy,
            "Dipole Energy": self.dipole_energy,
            "Nuclear Repulsion Energy": self.nuclear_repulsion_energy,
            "Dipole Moment": self.dipole_moment.tolist() if self.dipole_moment is not None else None,
            "Nuclear Dipole Moment": self.nuclear_dipole_moment.tolist() if self.nuclear_dipole_moment is not None else None,
            "RHF Dipole Moment": self.rhf_dipole_moment.tolist() if self.rhf_dipole_moment is not None else None,
            "Quadrupole Moment": self.quadrupole_moment.tolist() if self.quadrupole_moment is not None else None,
            "Orbital Energies": self.orbital_energies.tolist() if self.orbital_energies is not None else None
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

