import psi4
import numpy as np
import time
import json

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

    def run(self):
        # define molecule and options
        mol = psi4.geometry(self.molecule_string)
        psi4.set_options(self.psi4_options)
        self.rhf_energy, wfn = psi4.energy("scf", return_wfn=True)
        self.wfn = wfn

        mints = psi4.core.MintsHelper(wfn.basisset())
        ndocc = wfn.nalpha()
        C = np.asarray(wfn.Ca())
        Cocc = C[:, :ndocc]
        D = np.einsum("pi,qi->pq", Cocc, Cocc)

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

        l_dot_mu_el = sum(self.lambda_vector[i] * mu_ao[i] for i in range(3))

        mu_exp = np.array([
            2 * np.einsum("pq,pq->", mu_ao[i], D) for i in range(3)
        ]) + mu_nuc

        self.rhf_dipole_moment = mu_exp.copy()
        l_dot_mu_nuc = sum(self.lambda_vector[i] * mu_nuc[i] for i in range(3))
        l_dot_mu_exp = sum(self.lambda_vector[i] * mu_exp[i] for i in range(3))

        d_c = 0.5 * l_dot_mu_nuc**2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp**2
        self.dipole_energy = d_c

        Q_ao_xx, Q_ao_xy, Q_ao_xz, Q_ao_yy, Q_ao_yz, Q_ao_zz = [np.asarray(x) for x in mints.ao_quadrupole()]
        Q_ao = [Q_ao_xx, Q_ao_xy, Q_ao_xz, Q_ao_yy, Q_ao_yz, Q_ao_zz]

        Q_PF = -0.5 * self.lambda_vector[0]**2 * Q_ao_xx
        Q_PF -= 0.5 * self.lambda_vector[1]**2 * Q_ao_yy
        Q_PF -= 0.5 * self.lambda_vector[2]**2 * Q_ao_zz
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[1] * Q_ao_xy
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[2] * Q_ao_xz
        Q_PF -= self.lambda_vector[1] * self.lambda_vector[2] * Q_ao_yz

        d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el
        H_0 = T + V
        H = H_0 + Q_PF #+ d_PF

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
            J = np.einsum("pqrs,rs->pq", I, D)
            K = np.einsum("prqs,rs->pq", I, D)
            M = np.einsum("pq,rs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)
            N = np.einsum("pr,qs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)
            F = H + 2 * J - K - N # + 2 * M - N

            diis_e = A @ (F @ D @ S - S @ D @ F) @ A
            dRMS = np.sqrt(np.mean(diis_e**2))
            E_scf = np.einsum("pq,pq->", F + H, D) + Enuc #+ d_c

            if abs(E_scf - Eold) < E_conv and dRMS < D_conv:
                break
            Eold = E_scf

            Fp = A @ F @ A
            e, C2 = np.linalg.eigh(Fp)
            C = A @ C2
            Cocc = C[:, :ndocc]
            D = np.einsum("pi,qi->pq", Cocc, Cocc)

            mu_exp = np.array([
                2 * np.einsum("pq,pq->", mu_ao[i], D) for i in range(3)
            ]) + mu_nuc

            l_dot_mu_exp = sum(self.lambda_vector[i] * mu_exp[i] for i in range(3))
            d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el
            H = H_0 + Q_PF #+ d_PF
            d_c = 0.5 * l_dot_mu_nuc**2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp**2
        else:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

        self.cqed_rhf_energy = E_scf
        self.coefficients = C
        self.density_matrix = D
        self.fock_matrix = F
        self.orbital_energies = e
        self.dipole_moment = mu_exp
        self.nuclear_dipole_moment = mu_nuc

        q_exp = np.array([2 * np.einsum("pq,pq->", Q, D) for Q in Q_ao])
        self.quadrupole_moment = q_exp

        wfn_dict = psi4.core.Wavefunction.to_file(wfn)
        wfn_dict['matrix']['Ca'] = np.copy(C)
        wfn_dict['matrix']['Cb'] = np.copy(C)
        wfn_dict['matrix']['Da'] = np.copy(D)
        wfn_dict['matrix']['Db'] = np.copy(D)
        wfn_dict['matrix']['Fa'] = np.copy(F)
        wfn_dict['matrix']['Fb'] = np.copy(F)
        wfn_dict['vector']['epsilon_a'] = np.copy(e)
        wfn_dict['vector']['epsilon_b'] = np.copy(e)
        self.qed_wfn = psi4.core.Wavefunction.from_file(wfn_dict)

    def summary(self):
        print(f"RHF Energy:           {self.rhf_energy:.8f} Ha")
        print(f"CQED-RHF Energy:      {self.cqed_rhf_energy:.8f} Ha")
        print(f"Dipole Energy:        {self.dipole_energy:.8f} Ha")
        print(f"Nuclear Repulsion:    {self.nuclear_repulsion_energy:.8f} Ha")
        print(f"Dipole Moment:        {self.dipole_moment}")

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

