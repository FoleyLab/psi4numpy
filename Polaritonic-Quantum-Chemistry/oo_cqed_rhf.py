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

        # electronic dipole moment in SO basis
        self.dipole_so_x = np.asarray(mints.so_dipole()[0])
        self.dipole_so_y = np.asarray(mints.so_dipole()[1])
        self.dipole_so_z = np.asarray(mints.so_dipole()[2])



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

            F_can = H_0 + 2 * J - K 

             

        else:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")
        
        # update the dipole expectation value with the converged density matrix
        mu_exp = np.array([
            2 * oe.contract("pq,pq->", mu_ao[i], D, optimize='optimal') for i in range(3)
        ]) + mu_nuc

        # compute the quadrupole contribution to the energy
        self.o_dse_energy = 2 * oe.contract("pq,pq->", Q_PF, D, optimize="optimal")

        self.K_dse_energy = -1 * oe.contract("pq,pq->", N, D, optimize="optimal") 

        # compute the RHF energy without the DSE terms
        self.rhf_energy_no_cav = oe.contract("pq,pq->", F_can + H_0, D, optimize="optimal") + Enuc 

        # update d_exp
        d_exp = sum(self.lambda_vector[i] * mu_exp[i] for i in range(3))
        d_c = 0.5 * d_nuc**2 - d_nuc * d_exp + 0.5 * d_exp**2
        self.dipole_energy = d_c
        d_PF = (d_nuc - d_exp) * d_ao

        # go ahead and grab the dipole and quadrupole moment gradients for later use
        c_origin = [0.0, 0.0, 0.0]
        max_order = 2

        # symmetrization step
        D = 0.5 * (D + oe.contract("rs->sr", D, optimize="optimal"))
        Dp4 = psi4.core.Matrix.from_array( D )
        
        self.multipole_grad = np.asarray(mints.multipole_grad(Dp4, max_order, c_origin))


        self.cqed_rhf_energy = E_scf
        self.coefficients = C
        self.density_matrix = D
        self.fock_matrix = F
        self.canonical_fock_matrix = F_can
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

                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[0] ** 2 * self.multipole_grad[deriv_index, 3]
                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[1] ** 2 * self.multipole_grad[deriv_index, 6]
                self.o_dse_gradient[deriv_index] -= 0.5 * self.lambda_vector[2] ** 2 * self.multipole_grad[deriv_index, 8]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[0] * self.lambda_vector[1] * self.multipole_grad[deriv_index, 4]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[0] * self.lambda_vector[2] * self.multipole_grad[deriv_index, 5]
                self.o_dse_gradient[deriv_index] -= self.lambda_vector[1] * self.lambda_vector[2] * self.multipole_grad[deriv_index, 7]

        print(self.o_dse_gradient.reshape(self.num_atoms, 3))


    def calc_dipole_dipole_gradient(self):
        """ Calculate the dipole-dipole gradient using the CQED-RHF results.
        Returns:
            numpy.ndarray: The dipole-dipole gradient as a numpy array.
        """
        # instantiate mints
        mints = psi4.core.MintsHelper(self.psi4_wfn.basisset())

        self.K_dse_gradient = np.zeros(3 * self.num_atoms)

        # get the x, y, and z components of exchange contribution
        Da = self.density_matrix
        Db = self.density_matrix

        # D(p,q) = - mu(r,s) [ Da(p,r)Da(s,q) + Db(p,r) Da(s,q) ] ####
        # these are the density matrices contracted with the different dipole components
        tmp_a_x = -oe.contract("rs, pr, sq -> pq", self.dipole_so_x, Da, Da, optimize="optimal")
        tmp_a_y = -oe.contract("rs, pr, sq -> pq", self.dipole_so_y, Da, Da, optimize="optimal")
        tmp_a_z = -oe.contract("rs, pr, sq -> pq", self.dipole_so_z, Da, Da, optimize="optimal")
        tmp_b_x = -oe.contract("rs, pr, sq -> pq", self.dipole_so_x, Db, Da, optimize="optimal")
        tmp_b_y = -oe.contract("rs, pr, sq -> pq", self.dipole_so_y, Db, Da, optimize="optimal")
        tmp_b_z = -oe.contract("rs, pr, sq -> pq", self.dipole_so_z, Db, Da, optimize="optimal")

        # sum together the components 
        D_x = tmp_a_x + tmp_b_x
        D_y = tmp_a_y + tmp_b_y
        D_z = tmp_a_z + tmp_b_z

        # symmetrize D
        D_x = 0.5 * (D_x + oe.contract("rs->sr", D_x, optimize="optimal"))
        D_y = 0.5 * (D_y + oe.contract("rs->sr", D_y, optimize="optimal"))
        D_z = 0.5 * (D_z + oe.contract("rs->sr", D_z, optimize="optimal"))

        # cast as psi4 arrays
        Dp4_x = psi4.core.Matrix.from_array(D_x)
        Dp4_y = psi4.core.Matrix.from_array(D_y)
        Dp4_z = psi4.core.Matrix.from_array(D_z)

        # get the dipole-dipole gradient
        tmp_x = np.asarray(mints.dipole_grad(Dp4_x))
        tmp_y = np.asarray(mints.dipole_grad(Dp4_y))
        tmp_z = np.asarray(mints.dipole_grad(Dp4_z))

        # loop over atoms
        for atom in range(self.num_atoms):
            for cart in range(3):
                deriv_index = 3 * atom + cart

                # calculate the gradient components
                self.K_dse_gradient[deriv_index] += self.lambda_vector[2] ** 2 * tmp_z[deriv_index, 2]

    
    def calc_dipole_dipole_gradient_2(self):
        """
        NEEDS COMPLETING: Method to compute the two-electron integral gradient terms

        Arguments
        ---------
        geometry_string : str
            psi4 molecule string

        basis_set : str
            basis set to use for the calculation, defaults to 'sto-3g'

        method : str
            quantum chemistry method to use for the calculation, defaults to 'scf'

        The two-electron integral gradient terms are the derivatives of the two-electron integrals with respect to the nuclear coordinates.
        To compute the two-electron integral gradient terms, we need the two-electron integrals and the nuclear repulsion gradient.
        We will get these from a converged Hartree-Fock calculation.
        """
        # instantiate mints
        mints = psi4.core.MintsHelper(self.psi4_wfn.basisset())

        # initialize the two-electron integrals derivative matrices
        n_atoms = self.num_atoms
        n_orbitals = self.n_orbitals

        # initialize three arrays for the K_dse terms
        d_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        d_matrix = np.zeros((n_orbitals, n_orbitals))
        K_dse_deriv = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        K_dse_gradient = np.zeros(3 * n_atoms)

        # get the dipole integral derivatives
        #dipole_derivs = np.asarray(mints.ao_elec_dip_deriv1())
        

        # get the dipole integrals
        d_matrix = self.lambda_vector[0] * np.asarray(mints.ao_dipole()[0])
        print("x contribution of d matrix")
        print(d_matrix)
        d_matrix += self.lambda_vector[1] * np.asarray(mints.ao_dipole()[1])
        print("y contribution of d matrix")
        print(d_matrix)
        d_matrix += self.lambda_vector[2] * np.asarray(mints.ao_dipole()[2])
        print("z contribution of d matrix")
        print(d_matrix)

        #print("The shape of dipole_derivs is ",np.shape(dipole_derivs))
        print("The shape of d_matrix is ", np.shape(d_matrix))


        # need to think if this should be multiplied by 2 for alpha and beta!!!                                    
        D = self.density_matrix

        # loop over all of the atoms
        for atom_index in range(n_atoms):
            # Derivatives with respect to x, y, and z of the current atom
            _dip_deriv = np.asarray(mints.ao_elec_dip_deriv1(atom_index))
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                # get element of d_deriv:
                if cart_index == 0:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[0] + self.lambda_vector[1] * _dip_deriv[3] + self.lambda_vector[2] * _dip_deriv[6]

                elif cart_index == 1:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[1] + self.lambda_vector[1] * _dip_deriv[4] + self.lambda_vector[2] * _dip_deriv[7]

                else:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[2] + self.lambda_vector[1] * _dip_deriv[5] + self.lambda_vector[2] * _dip_deriv[8]

            

                # add code to contract d_derivs * d_matrix with D to get K_deriv, K^dse_uv = -1 sum_ls * d'_us * dlv D_ls
                K_dse_deriv[deriv_index] = -1 * oe.contract("us,lv,ls->uv", d_derivs[deriv_index, :, :], d_matrix, D, optimize="optimal")

                K_dse_gradient[deriv_index] = oe.contract("uv, uv->", D, K_dse_deriv[deriv_index, :, :], optimize="optimal")

        # add code to return the J_gradient and K_gradient
        return K_dse_gradient


    def calc_numerical_gradient(self, delta=1.0e-5):
        """Calculate the numerical gradient of the CQED-RHF energy with respect to the lambda vector.
        
        This method uses finite differences to compute the gradient.
        
        Args:
            delta (float): The step size for finite differences. Default is 1.0e-5.
        
        Returns:
            numpy.ndarray: The numerical gradient as a numpy array.
        """
        
        # copy original molecule string
        original_molecule_string = self.molecule_string
        self.numerical_energy_gradient = np.zeros(self.num_atoms * 3)
        self.numerical_o_dse_gradient = np.zeros(self.num_atoms * 3)
        self.numerical_K_dse_gradient = np.zeros(self.num_atoms * 3)
        self.numerical_scf_gradient = np.zeros(self.num_atoms * 3)

        # converstion from Angstroms to Bohr
        ang_to_Bohr = 1 / 0.52917721092  # convert Angstroms to Bohr


        for i in range(self.num_atoms):
            for j in range(3):
                _displacement = np.zeros((self.num_atoms, 3))
                _displacement[i, j] = delta
                self.molecule_string = self.modify_geometry_string(original_molecule_string, _displacement)
                self.calc_cqed_rhf_energy()
                energy_plus = self.cqed_rhf_energy
                o_dse_plus = self.o_dse_energy
                K_dse_plus = self.K_dse_energy
                scf_en_plus = self.rhf_energy_no_cav

                self.molecule_string = self.modify_geometry_string(original_molecule_string, -_displacement)
                self.calc_cqed_rhf_energy()
                energy_minus = self.cqed_rhf_energy
                o_dse_minus = self.o_dse_energy
                K_dse_minus = self.K_dse_energy
                scf_en_minus = self.rhf_energy_no_cav

                self.numerical_energy_gradient[i * 3 + j] = (energy_plus - energy_minus) / (2 * delta * ang_to_Bohr)
                self.numerical_o_dse_gradient[i * 3 + j] = (o_dse_plus - o_dse_minus) / (2 * delta * ang_to_Bohr)
                self.numerical_K_dse_gradient[i * 3 + j] = (K_dse_plus - K_dse_minus) / (2 * delta * ang_to_Bohr)
                self.numerical_scf_gradient[i * 3 + j] = (scf_en_plus - scf_en_minus) / (2 * delta * ang_to_Bohr)

    def modify_geometry_string(self, geometry_string, displacement_array):
        """
        Extracts Cartesian coordinates from a Psi4 geometry string, applies a
        transformation function to the coordinates, and returns a new geometry string.

        Args:
            geometry_string (str): A Psi4 molecular geometry string.
            transformation_function (callable): A function that takes a NumPy
                array of Cartesian coordinates (N x 3) as input and returns a
                NumPy array of the same shape with the transformed coordinates.

        Returns:
            str: A new Psi4 molecular geometry string with the transformed coordinates.
        """
        lines = geometry_string.strip().split('\n')
        atom_data = []
        symmetry = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("symmetry"):
                symmetry = line
                continue
            parts = line.split()
            if len(parts) == 4:
                atom = parts[0]
                try:
                    x, y, z = map(float, parts[1:])
                    atom_data.append([atom, x, y, z])
                except ValueError:
                    # Handle cases where the line might not be atom coordinates
                    pass
        if not atom_data:
            return ""

        coordinates = np.array([[data[1], data[2], data[3]] for data in atom_data])

        # Apply the transformation function
        transformed_coordinates = displacement_array  + coordinates

        new_geometry_lines = []
        for i, data in enumerate(atom_data):
            atom = data[0]
            new_geometry_lines.append(f"{atom} {transformed_coordinates[i, 0]:.8f} {transformed_coordinates[i, 1]:.8f} {transformed_coordinates[i, 2]:.8f}")

        new_geometry_string = "\n".join(new_geometry_lines)
        #if symmetry:
        #    new_geometry_string += f"\n{symmetry}"

        # Add the "no_reorient" and "no_com" lines
        new_geometry_string += "\nno_reorient\nno_com"
        # Add the "symmetry c1" line
        new_geometry_string += "\nsymmetry c1"

        return f"""{new_geometry_string}"""
    


    def compute_nuclear_repulsion_gradient(self):
        """
        Method to compute the nuclear repulsion gradient

        Arguments
        ---------
        geometry_string : str
            psi4 molecule string

        The nuclear repulsion gradient only depends on the atom identities and positions
        """
        # Define your molecular geometry
        molecule = psi4.geometry(self.molecule_string)

        # get the nuclear repulsion gradient
        self.nuclear_repulsion_gradient = np.asarray(molecule.nuclear_repulsion_energy_deriv1())


    def compute_fock_matrix_term(self):
        """
        Method to compute the Fock matrix

        Arguments
        ---------

        geometry_string : str
            psi4 molecule string

        basis_set : str
            basis set to use for the calculation, defaults to 'sto-3g'

        method : str
            quantum chemistry method to use for the calculation, defaults to 'scf'

        The Fock matrix is the matrix representation of the Fock operator, which is used in Hartree-Fock calculations.
        To compute the Fock matrix in the MO basis, we need the one-electron and two-electron integrals and the density matrix.
        We will get these from a converged Hartree-Fock calculation.
        """
        # instantiate mints
        mints = psi4.core.MintsHelper(self.psi4_wfn.basisset())

        # get the number of atoms and orbitals
        n_atoms = self.num_atoms
        n_orbitals = self.n_orbitals
        n_docc = self.ndocc

        # need to store the coefficients as a psi4 matrix
        C = psi4.core.Matrix.from_array(self.coefficients)


        # now compute the overlap gradient and contract with Fock matrix
        overlap_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.canonical_overlap_gradient = np.zeros(3 * n_atoms) # uses the canonical Fock matrix
        self.overlap_gradient = np.zeros(3 * n_atoms) # uses the Fock matrix with QED terms


        for atom_index in range(n_atoms):

            # Derivatives with respect to x, y, and z of the current atom
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                # Get overlap derivatives for this atom and Cartesian component
                overlap_derivs[deriv_index, :, :] = np.asarray(mints.mo_oei_deriv1("OVERLAP", atom_index, C, C)[cart_index])
                self.canonical_overlap_gradient[deriv_index] = -2.0 * oe.contract('ii,ii->', self.canonical_fock_matrix[:n_docc, :n_docc], overlap_derivs[deriv_index, :n_docc, :n_docc], optimize='optimal')
                self.overlap_gradient[deriv_index] = -2.0 * oe.contract('ii,ii->', self.fock_matrix[:n_docc, :n_docc], overlap_derivs[deriv_index, :n_docc, :n_docc], optimize='optimal') 


    def compute_one_electron_integral_gradient_terms(self):
        """
        Method to compute the derivatives of the T and V integrals

        Arguments
        ---------
        geometry_string : str
            psi4 molecule string

        basis_set : str
            basis set to use for the calculation, defaults to 'sto-3g'

        method : str
            quantum chemistry method to use for the calculation, defaults to 'scf'

        The one-electron integral gradient terms are the derivatives of the one-electron integrals with respect to the nuclear coordinates.
        To compute the one-electron integral gradient terms, we need the one-electron integrals and the nuclear repulsion gradient.
        We will get these from a converged Hartree-Fock calculation.
        """

        # instantiate mints
        mints = psi4.core.MintsHelper(self.psi4_wfn.basisset())

        # get the number of atoms and orbitals
        n_atoms = self.num_atoms
        n_orbitals = self.n_orbitals

  
        # initialize the one-electron integrals derivative matrices
        kinetic_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        potential_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        

        self.kinetic_gradient = np.zeros(3 * n_atoms)
        self.potential_gradient = np.zeros(3 * n_atoms)

        # loop over all of the atoms
        for atom_index in range(n_atoms):
            # Derivatives with respect to x, y, and z of the current atom
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                # get the one-electron integral derivatives
                kinetic_derivs[deriv_index] = np.asarray(mints.ao_oei_deriv1("KINETIC", atom_index)[cart_index])
                potential_derivs[deriv_index] = np.asarray(mints.ao_oei_deriv1("POTENTIAL", atom_index)[cart_index])

                # add code to contract kinetic_derivs with D, multiply by 2 since self.density_matrix is alpha only
                self.kinetic_gradient[deriv_index] = 2 * oe.contract("uv,uv->", self.density_matrix, kinetic_derivs[deriv_index, :, :], optimize='optimal')

                # add code to contract potential_derivs with D
                self.potential_gradient[deriv_index] = 2 * oe.contract("uv,uv->", self.density_matrix, potential_derivs[deriv_index, :, :], optimize='optimal')


    def compute_two_electron_integral_gradient_terms(self):
        """
        NEEDS COMPLETING: Method to compute the two-electron integral gradient terms

        Arguments
        ---------
        geometry_string : str
            psi4 molecule string

        basis_set : str
            basis set to use for the calculation, defaults to 'sto-3g'

        method : str
            quantum chemistry method to use for the calculation, defaults to 'scf'

        The two-electron integral gradient terms are the derivatives of the two-electron integrals with respect to the nuclear coordinates.
        To compute the two-electron integral gradient terms, we need the two-electron integrals and the nuclear repulsion gradient.
        We will get these from a converged Hartree-Fock calculation.
        """
        # get the number of orbitals and the number of doubly occupied orbitals
        n_orbitals = self.n_orbitals
        n_atoms = self.num_atoms
        n_docc = self.ndocc



        # define D = 2 * sum_i C_pi * C_qi
        D = 2 * self.density_matrix

        # instantiate the MintsHelper object
        mints = psi4.core.MintsHelper(self.psi4_wfn.basisset())

        # initialize the two-electron integrals derivative matrices
        eri_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals, n_orbitals, n_orbitals))
        J_deriv = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        K_deriv = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        self.J_gradient = np.zeros(3 * n_atoms)
        self.K_gradient = np.zeros(3 * n_atoms)


        # loop over all of the atoms
        for atom_index in range(n_atoms):
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index


                # get the two-electron integral derivatives
                eri_derivs[deriv_index] = np.asarray(mints.ao_tei_deriv1(atom_index)[cart_index])

                # add code to contract eri_derivs with D to get J_deriv. J_uv = 2 * sum_ls (uv|ls) D_ls 
                # note we are factoring the 2 into D here
                J_deriv[deriv_index] = oe.contract("uvls,ls->uv", eri_derivs[deriv_index, :, :, :, :], D, optimize="optimal")

                # add code to contract eri_derivs with D to get K_deriv. K_uv = -1 * sum_ls (ul|vs) D_ls
                # note we are factoring the 2 into D here
                K_deriv[deriv_index] = -1 / 2 * oe.contract("ulvs,ls->uv", eri_derivs[deriv_index, :, :, :, :], D, optimize="optimal")

                # add code to contract J_deriv with D to get J_gradient
                self.J_gradient[deriv_index] = 1 / 2 * oe.contract("uv,uv->", D, J_deriv[deriv_index, :, :], optimize="optimal")

                # add code to contract K_deriv with D to get K_gradient
                self.K_gradient[deriv_index] = 1 / 2 * oe.contract("uv,uv->", D, K_deriv[deriv_index, :, :], optimize="optimal")

    def compute_canonical_gradients(self):
        self.compute_fock_matrix_term()
        self.compute_one_electron_integral_gradient_terms()
        self.compute_two_electron_integral_gradient_terms()
        self.compute_nuclear_repulsion_gradient()

        # sum together the different contributions to get the total canonical gradient
        self.canonical_gradient = (self.kinetic_gradient + self.potential_gradient + self.J_gradient + self.K_gradient).reshape(3,3) + self.nuclear_repulsion_gradient

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
