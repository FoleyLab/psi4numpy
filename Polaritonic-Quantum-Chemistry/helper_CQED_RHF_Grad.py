"""
Reference implementation of the CQED-RHF analytic nuclear gradient.

"""

import psi4
import numpy as np
from helper_CQED_RHF import *

class QEDRHFGrad:
    def __init__(self, options):
        """
        Initialize the RHFGrad object with a configuration dictionary.

        Parameters:
        options (dict): A dictionary containing key details for computation.

        Keys 
        ----
        molecule_string (str): 
            The molecular structure in string format, including charge and multiplicity and symmetry c1

        psi4_options_dict (dict): 
            A dictionary of options for Psi4 including basis set, convergence, scf type 'pk'

        lambda_vector (list): 
            A list of lambda values for the cavity (if needed)
            
        """
        if not isinstance(options, dict):
            raise ValueError("Config must be a dictionary.")
        
        required_keys = ["molecule_string", "psi4_options_dict", "lambda_vector"]
        for key in required_keys:
            if key not in options:
                raise ValueError(f"Missing required key: {key}")
            
        # capture the options dictionary values as attributes 
        self.molecule_string = options["molecule_string"]
        self.psi4_options = options["psi4_options_dict"]
        self.lambda_vector = options["lambda_vector"]



    def compute_qed_rhf_wfn(self):
        """
        Method to compute the RHF wavefunction using Psi4 and return the wavefunction object.

        Attributes
        ----------
        molecule_string (str): 
            The molecular structure in string format, including charge and multiplicity and symmetry c1

        psi4_options (dict):
            A dictionary of options for Psi4 including basis set, convergence, scf type 'pk'

        Returns
        -------
        psi4.core.Wavefunction: 
            The wavefunction object containing the computed RHF wavefunction.
        """
        molecule = psi4.geometry(self.molecule_string)
        psi4.set_options(self.psi4_options)

        # Compute the RHF wavefunction
        rhf_e, rhf_wfn = psi4.energy("SCF", return_wfn=True, molecule=molecule)
        if rhf_wfn is None:
            raise ValueError("Failed to compute RHF wavefunction.")
        
        self.rhf_energy = rhf_e 

        # now get the cqed info 
        # compute the QED-RHF energy and density matrix
        cqed_dict = cqed_rhf(self.lambda_vector, self.molecule_string, self.psi4_options)

        # parse dictionary for ordinary RHF and CQED-RHF energy
        _rhf_e = cqed_dict["RHF ENERGY"]
        _cqed_rhf_e = cqed_dict["CQED-RHF ENERGY"]

        # confirm the rhf energy from this method mmatches psi4
        assert np.isclose(_rhf_e, rhf_e)

        # parse dictionary for density matrix
        _cqed_rhf_D = cqed_dict["CQED-RHF DENSITY MATRIX"]
        _cqed_rhf_C = cqed_dict["CQED-RHF C"]
        _cqed_rhf_eps = cqed_dict["CQED-RHF EPS"]
        _cqed_rhf_F = cqed_dict["CQED-RHF FOCK"]

        # update the wavefunction object with the QED-RHF information
        # update the wfn object with the coefficients and density matrix from the cavity calculation
        wfn_dict = psi4.core.Wavefunction.to_file(rhf_wfn)

        # now update the quantities with cqed quantities
        wfn_dict['matrix']['Ca'] = np.copy(_cqed_rhf_C)
        wfn_dict['matrix']['Cb'] = np.copy(_cqed_rhf_C)
        wfn_dict['matrix']['Da'] = np.copy(_cqed_rhf_D)
        wfn_dict['matrix']['Db'] = np.copy(_cqed_rhf_D)
        wfn_dict['matrix']['Fa'] = np.copy(_cqed_rhf_F)
        wfn_dict['matrix']['Fb'] = np.copy(_cqed_rhf_F)
        wfn_dict['vector']['epsilon_a'] = np.copy(_cqed_rhf_eps)
        wfn_dict['vector']['epsilon_b'] = np.copy(_cqed_rhf_eps)
        # push these back to the wavefunction object
        self.qed_rhf_wfn = psi4.core.Wavefunction.from_file(wfn_dict)


        return self.qed_rhf_wfn
    

    def compute_gradient_quantities(self):
        """
        Method to compute the necessary integrals for the RHF gradient, specifically the terms in this equation:
        .. math::
           \frac{\partial E}{\partial x_i} = \frac{\partial E_{nuc}}{\partial x_i} + \sum_{\mu\nu} D_{\mu\nu} \frac{\partial h_{\mu\nu}}{\partial x_i} 
           + \frac{1}{2} \sum_{\mu\nu\lambda\sigma} D_{\mu\nu} D_{\lambda\sigma} \frac{\partial (\mu\nu|\lambda\sigma)}{\partial x_i}
           - \sum_{pq} F_{pq} \frac{\partial S_{pq}}{\partial R_A}

        

        Attributes
        ----------
        molecule_string (str):
            The molecular structure in string format, including charge and multiplicity and symmetry c1

        psi4_options (dict):
            A dictionary of options for Psi4 including basis set, convergence, scf type 'pk'

        After Exectution
        ----------------

        fock_matrix_mo (numpy.ndarray):
            The Fock matrix in the molecular orbital basis.

        density_matrix (numpy.ndarray): 
            The density matrix in the molecular orbital basis.

        overlap_deriv_matrix_ao (numpy.ndarray):
            The overlap derivative matrix in the atomic orbital basis.

        potential_deriv_matrix_ao (numpy.ndarray):
            The potential derivative matrix in the atomic orbital basis.

        kinetic_deriv_matrix_ao (numpy.ndarray):
            The kinetic derivative matrix in the atomic orbital basis.

        J_deriv_matrix_ao (numpy.ndarray):
            The J derivative matrix in the atomic orbital basis.

        K_deriv_matrix_ao (numpy.ndarray):
            The K derivative matrix in the atomic orbital basis.

        nuclear_gradient (numpy.ndarray):
            The nuclear gradient

        """
        # define the molecule
        molecule = psi4.geometry(self.molecule_string)

        # get number of atoms
        n_atoms = molecule.natom()

        # get the nuclear gradient as a 1D numpy array
        self.nuclear_energy_gradient =  np.asarray(molecule.nuclear_repulsion_energy_deriv1()).flatten()

        # Get the RHF wavefunction
        wfn = self.compute_qed_rhf_wfn()

        # get the number of orbitals and the number of doubly occupied orbitals
        n_orbitals = wfn.nmo()
        n_docc = wfn.nalpha()

        # get the orbital transformation matrix
        C = wfn.Ca() # -> as psi4 matrix object
        Cnp = np.asarray(C) # -> as numpy array

        # get the Density matrix by summing over the occupied orbital transformation matrix
        Cocc = Cnp[:, :n_docc]
        self.density_matrix = np.einsum("pi,qi->pq", Cocc, Cocc)
        
        # get Da and Db from wfn object -> might be redundant
        Da = np.asarray(wfn.Da())
        Db = np.asarray(wfn.Db())

        # D symmetrized
        D_sym = 0.5 * (Da + Db) + 0.5 * np.einsum('rs -> sr', (Da + Db))

        D = psi4.core.Matrix.from_array(D_sym)

        # origin vector
        origin = [0.0, 0.0, 0.0]

        # instantiate the MintsHelper object
        mints = psi4.core.MintsHelper(wfn.basisset())

        # get the derivatives of the dipole integrals
        mu_mo = np.asarray(mints.dipole_grad(D))
        print("Shape of mu_mo: ", mu_mo.shape)

        # get the derivatives of the quadrupole integrals
        max_order = 2
        q_mo = np.asarray(mints.multipole_grad(D, max_order, origin))

        print("Shape of q_mo: ", q_mo.shape)

        # get the one-electron integrals
        H_ao = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

        # transform H_ao to the MO basis
        H_mo = np.einsum('uj, vi, uv', Cnp, Cnp, H_ao)

        # get the two-electron integrals, use psi4 to transform into the MO basis because that is more efficient
        ERI =  np.asarray(mints.mo_eri(C, C, C, C))

        # Build the Fock matrix
        F = H_mo + 2 * np.einsum("ijkk->ij", ERI[:, :, :n_docc, :n_docc]) 
        F -= np.einsum("ikkj->ij", ERI[:, :n_docc, :n_docc, :] )

        self.fock_matrix_mo = np.copy(F)

        # initialize array for the integral derivatives
        self.overlap_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.potential_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.kinetic_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.quad_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.eri_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals, n_orbitals, n_orbitals))
        self.J_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.K_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.K_dse_deriv_matrix_ao = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))


        # initialize the gradient arrays
        self.pulay_force = np.zeros(3 * n_atoms)
        self.kinetic_gradient = np.zeros(3 * n_atoms)
        self.potential_gradient = np.zeros(3 * n_atoms)
        self.quad_gradient = np.zeros(3 * n_atoms)
        self.J_gradient = np.zeros(3 * n_atoms)
        self.K_gradient = np.zeros(3 * n_atoms)
        self.K_dse_gradient = np.zeros(3 * n_atoms)
        self.total_gradient = np.zeros(3 * n_atoms)

        # loop over the atoms
        for i in range(n_atoms):
            # loop over the cartesian coordinates
            for j in range(3):
                # define the derivative index
                deriv_index = 3 * i + j

                # get the one-electron integral derivatives
                # overlap is in the MO basis
                self.overlap_deriv_matrix_ao[deriv_index] = np.asarray(mints.mo_oei_deriv1("OVERLAP", i, C, C )[j])

                # all others are in the AO basis
                self.potential_deriv_matrix_ao[deriv_index] = np.asarray(mints.ao_oei_deriv1("POTENTIAL", i)[j])
                self.kinetic_deriv_matrix_ao[deriv_index] = np.asarray(mints.ao_oei_deriv1("KINETIC", i)[j])

                # get the two-electron integral derivatives
                self.eri_deriv_matrix_ao[deriv_index] = np.asarray(mints.ao_tei_deriv1(i)[j])

                # compute the J and K derivetives
                self.J_deriv_matrix_ao[deriv_index] = 2 * np.einsum("uvls,ls->uv", self.eri_deriv_matrix_ao[deriv_index, :, :, :, :], self.density_matrix)

                self.K_deriv_matrix_ao[deriv_index] =  -1 * np.einsum("ulvs,ls->uv", self.eri_deriv_matrix_ao[deriv_index, :, :, :, :], self.density_matrix)

                # now contract each of the derivatives with the density matrix to get the respective gradient components
                # Pulay force first
                self.pulay_force[deriv_index] = -2.0 * np.einsum("ii,ii->", self.fock_matrix_mo[:n_docc, :n_docc], self.overlap_deriv_matrix_ao[deriv_index, :n_docc, :n_docc])

                # kinetic gradient
                self.kinetic_gradient[deriv_index] = 2 * np.einsum("uv,uv->", self.density_matrix, self.kinetic_deriv_matrix_ao[deriv_index, :, :])

                # potential gradient
                self.potential_gradient[deriv_index] = 2 * np.einsum("uv,uv->", self.density_matrix, self.potential_deriv_matrix_ao[deriv_index, :, :])

                # J gradient
                self.J_gradient[deriv_index] = np.einsum("uv,uv->", self.density_matrix, self.J_deriv_matrix_ao[deriv_index, :, :])

                # K gradient
                self.K_gradient[deriv_index] = np.einsum("uv,uv->", self.density_matrix, self.K_deriv_matrix_ao[deriv_index, :, :])
                                                                    

        self.total_gradient = self.nuclear_energy_gradient + self.pulay_force + self.kinetic_gradient + self.potential_gradient + self.J_gradient + self.K_gradient

        return self.total_gradient
        
    
