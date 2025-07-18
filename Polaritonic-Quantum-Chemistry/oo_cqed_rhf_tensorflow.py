import psi4
import numpy as np
import time
import json
import opt_einsum as oe
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

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

        # instantiate MintsHelper, store instance as an attribute of the CQEDRHFCalculator class
        self.mints = psi4.core.MintsHelper(wfn.basisset())

        # get some standard information and also store as attributes
        ndocc = wfn.nalpha()
        self.ndocc = ndocc
        self.n_orbitals = wfn.nmo()
        self.num_atoms = mol.natom()
        
        # get initial density matrix
        C = np.asarray(wfn.Ca())
        Cocc = C[:, :ndocc]
        D = oe.contract("pi,qi->pq", Cocc, Cocc, optimize="optimal")

        # get standard integrals
        V = np.asarray(self.mints.ao_potential())
        T = np.asarray(self.mints.ao_kinetic())
        I = np.asarray(self.mints.ao_eri())

        # get nuclear dipole contributions
        mu_nuc_x = mol.nuclear_dipole()[0]
        mu_nuc_y = mol.nuclear_dipole()[1]
        mu_nuc_z = mol.nuclear_dipole()[2]

        # electronic dipole integrals in AO basis
        mu_ao_x = np.asarray(self.mints.ao_dipole()[0])
        mu_ao_y = np.asarray(self.mints.ao_dipole()[1])
        mu_ao_z = np.asarray(self.mints.ao_dipole()[2])

        #mu_nuc_x, mu_nuc_y, mu_nuc_z = mol.nuclear_dipole()
        mu_nuc = np.array([mu_nuc_x, mu_nuc_y, mu_nuc_z])

        #mu_ao_x, mu_ao_y, mu_ao_z = [np.asarray(x) for x in mints.ao_dipole()]
        mu_ao = np.array([mu_ao_x, mu_ao_y, mu_ao_z])

        d_ao = sum(self.lambda_vector[i] * mu_ao[i] for i in range(3))

        # electronic dipole moment expectation value from RHF 
        mu_exp = np.array([
            2 * oe.contract("pq,pq->", mu_ao[i], D, optimize='optimal') for i in range(3)
        ]) 
        # this is the current dipole moment from the RHF wavefunction, at the end
        # of scf iterations it will be the CQED-RHF dipole moment
        self.rhf_dipole_moment = mu_exp + mu_nuc

        # handle quadrupole contribution to O_DSE
        Q_ao_xx, Q_ao_xy, Q_ao_xz, Q_ao_yy, Q_ao_yz, Q_ao_zz = [np.asarray(x) for x in self.mints.ao_quadrupole()]
        Q_ao = [Q_ao_xx, Q_ao_xy, Q_ao_xz, Q_ao_yy, Q_ao_yz, Q_ao_zz]

        Q_PF = -0.5 * self.lambda_vector[0]**2 * Q_ao_xx
        Q_PF -= 0.5 * self.lambda_vector[1]**2 * Q_ao_yy
        Q_PF -= 0.5 * self.lambda_vector[2]**2 * Q_ao_zz
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[1] * Q_ao_xy
        Q_PF -= self.lambda_vector[0] * self.lambda_vector[2] * Q_ao_xz
        Q_PF -= self.lambda_vector[1] * self.lambda_vector[2] * Q_ao_yz

        # canonical core Hamiltonian
        H_0 = T + V

        # QED core Hamiltonian
        H = H_0 + Q_PF 

        S = self.mints.ao_overlap()
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.0e-16)
        A = np.asarray(A)

        Eold = 0.0
        Enuc = mol.nuclear_repulsion_energy()
        self.nuclear_repulsion_energy = Enuc

        maxiter = 500
        E_conv = self.psi4_options.get("e_convergence", 1.0e-7)
        D_conv = self.psi4_options.get("d_convergence", 1.0e-5)

        for scf_iter in range(1, maxiter + 1):

            # canonical J and K contributions
            J = oe.contract("pqrs,rs->pq", I, D, optimize="optimal")
            K = oe.contract("prqs,rs->pq", I, D, optimize="optimal")
            
            # K_dse contribution
            N = oe.contract("pr,qs,rs->pq", d_ao, d_ao, D, optimize="optimal")

            # updated Fock matrix
            F = H + 2 * J - K - N 

            diis_e = A @ (F @ D @ S - S @ D @ F) @ A
            dRMS = np.sqrt(np.mean(diis_e**2))

            # current QED-RHF energy, note D is only Da so we are performing einsum over (F+H) D
            E_scf = oe.contract("pq,pq->", F + H, D, optimize="optimal") + Enuc 

            if abs(E_scf - Eold) < E_conv and dRMS < D_conv:
                break
            Eold = E_scf

            Fp = A @ F @ A
            e, C2 = np.linalg.eigh(Fp)
            C = A @ C2
            Cocc = C[:, :ndocc]
            D = np.einsum("pi,qi->pq", Cocc, Cocc)
             

        # executes if number of SCF iterations hits the maximum without the break condition
        else:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")
        # store energy
        self.cqed_rhf_energy = E_scf
        # update the electronic dipole expectation value with the converged density matrix
        mu_exp = np.array([
            2 * oe.contract("pq,pq->", mu_ao[i], D, optimize='optimal') for i in range(3)
        ]) 

        # this is the full qed-rhf dipole moment, electronic + nuclear
        self.qedrhf_dipole_moment = mu_exp + mu_nuc

        self.nuclear_dipole_moment = mu_nuc
        self.qedrhf_electronic_dipole_moment = mu_exp

        # these can be useful for QED-CI
        self.d_nuc = sum(self.lambda_vector[i] * mu_nuc[i] for i in range(3)) # nuclear part of d
        self.d_exp_el = sum(self.lambda_vector[i] * mu_exp[i] for i in range(3)) # electronic part of <d>
        self.d_exp = self.d_exp_el + self.d_nuc # total <d>


        self.dipole_energy = 0.5 * self.d_nuc**2 - self.d_nuc * self.d_exp_el + 0.5 * self.d_exp_el**2

        # for qed-ci in photon number basis
        self.d_PF_pn = self.d_nuc * d_ao
        
        # for qed-ci in the coherent state basis
        self.d_PF_cs = - self.d_exp_el * d_ao

        self.coefficients = C
        self.density_matrix = D # just Da here!
        self.fock_matrix_ao = F
        self.canonical_fock_matrix_ao = H_0 + 2 * J - K
        self.canonical_fock_matrix_mo = C.T @ (H_0 + 2 * J - K)  @ C
        self.fock_matrix_mo = C.T @ F @ C
        self.orbital_energies = e
        self.dipole_moment = mu_exp
        self.nuclear_dipole_moment = mu_nuc
        self.Q_PF = Q_PF
        self.d_ao = d_ao

        # symmetrize the density matrix
        D = 0.5 * (self.density_matrix + oe.contract('rs->sr', self.density_matrix, optimize="optimal"))

        # cast as Psi4 matrix.  Also multiply by 2 to account for alpha and beta density matrices
        Dp4 = psi4.core.Matrix.from_array(2 * D)
        self.density_matrix_psi4 = Dp4


    def summary(self):
        print(f"RHF Energy:                    {self.rhf_energy:.8f} Ha")
        print(f"CQED-RHF Energy:               {self.cqed_rhf_energy:.8f} Ha")
        print(f"Dipole Energy:                 {self.dipole_energy:.8f} Ha")
        print(f"Nuclear Repulsion:             {self.nuclear_repulsion_energy:.8f} Ha")
        print(F"RHF Dipole Moment:             {self.rhf_dipole_moment}")
        print(f"CQED-RHF Dipole Moment:        {self.qedrhf_dipole_moment}")

    def compute_scf_gradient(self, qed_wfn=False):
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

        
    
    def hilber_quadrupole_gradient(self):
        
        
        C_origin = [0.0, 0.0, 0.0] # origin
        maxorder = 2 # quadrupole
        D = self.density_matrix * 2
        natom = self.num_atoms
        lambda_z = self.lambda_vector[2] # z-component of lambda vector
        
        # symmetrize D because dipole_grad only uses 1/2 the elements
        D = 0.5 * ( D + np.einsum('rs->sr',D) )
        D = psi4.core.Matrix.from_array(D)
        
        # 3N x 9 matrix of quadrupole derivatives
        quad_grad = np.asarray(self.mints.multipole_grad(D, maxorder, C_origin))
        
        # get requested component of quadrupole gradient
        zzdir = 8 # zz component

        # unpack zz-component 3N x 3 matrix (the 9th column)
        dse_gradient_zz = np.zeros((natom,3))
        for atom in range (0,natom):
            for cart in range (0,3):
                dse_gradient_zz[atom,cart] = quad_grad[atom*3+cart,zzdir]
        
        self.hilbert_o_dse = dse_gradient_zz * -0.5 * lambda_z * lambda_z

    
    def compute_quadrupole_gradient(self):
        """ Calculate the quadrupole gradient using the CQED-RHF results.
        Returns:
            numpy.ndarray: The quadrupole gradient as a numpy array.
        """

        # define c_origin as the origin of the coordinate system
        c_origin = [0.0, 0.0, 0.0]  # origin

        # define max_order = 2
        max_order = 2  # dipole and quadrupole

        # symmetrize the density matrix
        #D = 0.5 * (self.density_matrix + oe.contract('rs->sr', self.density_matrix, optimize="optimal"))

        # cast as Psi4 matrix.  Also multiply by 2 to account for alpha and beta density matrices
        #Dp4 = psi4.core.Matrix.from_array(2 * D)
        #self.density_matrix_psi4 = Dp4
        Dp4 = self.density_matrix_psi4
        # get the multipole gradient
        self.multipole_grad = np.asarray(self.mints.multipole_grad(Dp4, max_order, c_origin))

        # shape as (num_atoms, 3) array, like psi4 does
        self.o_dse_gradient = np.zeros((self.num_atoms,3))

        for atom_index in range(self.num_atoms):
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                self.o_dse_gradient[atom_index,cart_index] -= 0.5 * self.lambda_vector[0] ** 2 * self.multipole_grad[deriv_index, 3]
                self.o_dse_gradient[atom_index,cart_index] -= 0.5 * self.lambda_vector[1] ** 2 * self.multipole_grad[deriv_index, 6]
                self.o_dse_gradient[atom_index,cart_index] -= 0.5 * self.lambda_vector[2] ** 2 * self.multipole_grad[deriv_index, 8]
                self.o_dse_gradient[atom_index,cart_index] -= self.lambda_vector[0] * self.lambda_vector[1] * self.multipole_grad[deriv_index, 4]
                self.o_dse_gradient[atom_index,cart_index] -= self.lambda_vector[0] * self.lambda_vector[2] * self.multipole_grad[deriv_index, 5]
                self.o_dse_gradient[atom_index,cart_index] -= self.lambda_vector[1] * self.lambda_vector[2] * self.multipole_grad[deriv_index, 7]

        

    
    def compute_dipole_dipole_gradient(self):
        """
        Method to compute the two-electron integral gradient terms

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

        # initialize the two-electron integrals derivative matrices
        n_atoms = self.num_atoms
        n_orbitals = self.n_orbitals

        # initialize three arrays for the K_dse terms
        d_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        K_dse_deriv = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))

        # final gradient has shape (n_atoms, 3) like psi4
        self.K_dse_gradient = np.zeros((n_atoms, 3))

        d_matrix = self.d_ao

        # need to think if this should be multiplied by 2 for alpha and beta!!!                                    
        D = self.density_matrix

        # loop over all of the atoms
        for atom_index in range(n_atoms):
            # Derivatives with respect to x, y, and z of the current atom
            _dip_deriv = np.asarray(self.mints.ao_elec_dip_deriv1(atom_index))
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                # get element of d_deriv:
                if cart_index == 0:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[0] + self.lambda_vector[1] * _dip_deriv[3] + self.lambda_vector[2] * _dip_deriv[6]

                elif cart_index == 1:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[1] + self.lambda_vector[1] * _dip_deriv[4] + self.lambda_vector[2] * _dip_deriv[7]

                elif cart_index == 2:
                    d_derivs[deriv_index] += self.lambda_vector[0] * _dip_deriv[2] + self.lambda_vector[1] * _dip_deriv[5] + self.lambda_vector[2] * _dip_deriv[8]

                else:
                    raise ValueError("cart_index must be 0, 1, or 2.")
            

                # add code to contract d_derivs * d_matrix with D to get K_deriv, K^dse_uv = -1 sum_ls * d'_us * dlv D_ls
                K_dse_deriv[deriv_index] = -1 * oe.contract("us,lv,ls->uv", d_derivs[deriv_index, :, :], d_matrix, D, optimize="optimal")
                
                # we only performed this for Da Da, we need to multiply by 2 to account for the beta density matrix
                self.K_dse_gradient[atom_index, cart_index] = 2 * oe.contract("uv, uv->", D, K_dse_deriv[deriv_index, :, :], optimize="optimal")



    def compute_numerical_gradient(self, delta=1.0e-5):
        """Calculate the numerical gradient of the CQED-RHF energy with respect to the lambda vector.
        
        This method uses finite differences to compute the gradient.
        
        Args:
            delta (float): The step size for finite differences. Default is 1.0e-5.
        
        Returns:
            numpy.ndarray: The numerical gradient as a numpy array.
        """
        
        # copy original molecule string
        original_molecule_string = self.molecule_string

        # numerical gradient has shape (num_atoms, 3) like psi4
        self.numerical_energy_gradient = np.zeros((self.num_atoms, 3))


        # converstion from Angstroms to Bohr
        ang_to_Bohr = 1 / 0.52917721092  # convert Angstroms to Bohr


        for i in range(self.num_atoms):
            for j in range(3):

                # forward step
                _displacement = np.zeros((self.num_atoms, 3))
                _displacement[i, j] = delta
                self.molecule_string = self.modify_geometry_string(original_molecule_string, _displacement)
                self.calc_cqed_rhf_energy()
                energy_plus = self.cqed_rhf_energy

                # backward step
                self.molecule_string = self.modify_geometry_string(original_molecule_string, -_displacement)
                self.calc_cqed_rhf_energy()
                energy_minus = self.cqed_rhf_energy

                # compute gradient element
                self.numerical_energy_gradient[i, j] = (energy_plus - energy_minus) / (2 * delta * ang_to_Bohr)


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

        # get the nuclear repulsion gradient, which will have shape (num_atoms, 3)
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

        # get the number of atoms and orbitals
        n_atoms = self.num_atoms
        n_orbitals = self.n_orbitals
        n_docc = self.ndocc

        # need to store the coefficients as a psi4 matrix
        C = psi4.core.Matrix.from_array(self.coefficients)


        # now compute the overlap gradient and contract with Fock matrix
        overlap_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        

        # overlap gradient will have shape (n_atoms, 3) like psi4
        self.canonical_overlap_gradient = np.zeros((n_atoms, 3)) # uses the Fock matrix without QED terms
        self.overlap_gradient = np.zeros((n_atoms, 3)) # uses the Fock matrix with QED terms


        for atom_index in range(n_atoms):

            # Derivatives with respect to x, y, and z of the current atom
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                # Get overlap derivatives for this atom and Cartesian component
                overlap_derivs[deriv_index, :, :] = np.asarray(self.mints.mo_oei_deriv1("OVERLAP", atom_index, C, C)[cart_index])
                self.canonical_overlap_gradient[atom_index, cart_index] = -2.0 * oe.contract('ii,ii->', self.canonical_fock_matrix_mo[:n_docc, :n_docc], overlap_derivs[deriv_index, :n_docc, :n_docc], optimize='optimal')
                self.overlap_gradient[atom_index, cart_index] = -2.0 * oe.contract('ii,ii->', self.fock_matrix_mo[:n_docc, :n_docc], overlap_derivs[deriv_index, :n_docc, :n_docc], optimize='optimal') 


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

        # get the number of atoms and orbitals
        n_atoms = self.num_atoms
        n_orbitals = self.n_orbitals

  
        # initialize the one-electron integrals derivative matrices
        kinetic_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        potential_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))

        # gradients will have shape (n_atoms, 3) like psi4
        self.kinetic_gradient = np.zeros((n_atoms, 3))
        self.potential_gradient = np.zeros((n_atoms, 3))

        # loop over all of the atoms
        for atom_index in range(n_atoms):
            # Derivatives with respect to x, y, and z of the current atom
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index

                # get the one-electron integral derivatives
                kinetic_derivs[deriv_index] = np.asarray(self.mints.ao_oei_deriv1("KINETIC", atom_index)[cart_index])
                potential_derivs[deriv_index] = np.asarray(self.mints.ao_oei_deriv1("POTENTIAL", atom_index)[cart_index])

                # add code to contract kinetic_derivs with D, multiply by 2 since self.density_matrix is alpha only
                self.kinetic_gradient[atom_index, cart_index] = 2 * oe.contract("uv,uv->", self.density_matrix, kinetic_derivs[deriv_index, :, :], optimize='optimal')

                # add code to contract potential_derivs with D
                self.potential_gradient[atom_index, cart_index] = 2 * oe.contract("uv,uv->", self.density_matrix, potential_derivs[deriv_index, :, :], optimize='optimal')


    def compute_two_electron_integral_gradient_terms_2(self):
        n_orbitals = self.n_orbitals
        n_atoms = self.num_atoms

        #D_psi4 = self.density_matrix_psi4 # Use the psi4.core.Matrix version of D
        D_np = self.density_matrix * 2
        D_tf = tf.convert_to_tensor(D_np, dtype=tf.float64)  # use float64 if needed



        self.J_gradient = np.zeros((n_atoms, 3))
        self.K_gradient = np.zeros((n_atoms, 3))

        # --- REVISED PART 1: Getting ERI derivatives without NumPy conversion (initially) ---
        # Store eri_derivs as a list of lists of psi4.core.Matrix objects
        # eri_derivs_psi4[atom_idx][cart_idx] will be a psi4.core.Matrix of shape (n_orb^2, n_orb^2)

        # eri derivatives in one numpy array
        #eri_derivs_np = np.zeros((n_atoms, 3, n_orbitals, n_orbitals, n_orbitals, n_orbitals))
        #J_deriv_np = np.zeros((n_atoms, 3, n_orbitals, n_orbitals))
        #K_deriv_np = np.zeros((n_atoms, 3, n_orbitals, n_orbitals))

        #for atom_index in range(n_atoms):
        #    # ao_tei_deriv1 returns a list of 3 psi4.core.Matrix objects (X, Y, Z)
        #    eri_derivs_np[atom_index, :, :, :, :, :] = np.asarray(self.mints.ao_tei_deriv1(atom_index))
        #    J_deriv_np[atom_index, :, :, :] = oe.contract("xuvls,ls->xuv", eri_derivs_np[atom_index, :, :, :, :], D_np, optimize="optimal")
        #    K_deriv_np[atom_index, :, :, :] = -0.5 * oe.contract("xulvs,ls->xuv", eri_derivs_np[atom_index, :, :, :, :], D_np, optimize="optimal")
        #    self.J_gradient[atom_index, :] = 0.5 * oe.contract("uv,xuv->x", D_np, J_deriv_np[atom_index, :, :, :], optimize="optimal")
        #    self.K_gradient[atom_index, :] = 0.5 * oe.contract("uv,xuv->x", D_np, K_deriv_np[atom_index, :, :, :], optimize="optimal")
        for atom_index in range(n_atoms):
            eri_deriv = np.asarray(self.mints.ao_tei_deriv1(atom_index))  # shape (3, n, n, n, n)
            
            # convert to tensorflow array 
            eri_tf = tf.convert_to_tensor(eri_deriv, dtype=tf.float64)

            J_grad = 0.5 * tf.einsum("uv,xuvls,ls->x", D_tf, eri_tf, D_tf)
            K_grad = -0.25 * tf.einsum("uv,xulvs,ls->x", D_tf, eri_tf, D_tf)

            self.J_gradient[atom_index, :] = J_grad.numpy()
            self.K_gradient[atom_index, :] = K_grad.numpy()

            #self.J_gradient[atom_index, :] = 0.5 * oe.contract(
            #    "uv,xuvls,ls->x", D_np, eri_deriv, D_np, optimize="optimal"
            #)

            #self.K_gradient[atom_index, :] = -0.25 * oe.contract(
            #    "uv,xulvs,ls->x", D_np, eri_deriv, D_np, optimize="optimal"
            #)


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


        # define D = 2 * sum_i C_pi * C_qi
        D = 2 * self.density_matrix


        # initialize the two-electron integrals derivative matrices
        eri_derivs = np.zeros((3 * n_atoms, n_orbitals, n_orbitals, n_orbitals, n_orbitals))
        J_deriv = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))
        K_deriv = np.zeros((3 * n_atoms, n_orbitals, n_orbitals))

        # gradients will be shape (n_atoms, 3) like psi4
        self.J_gradient = np.zeros((n_atoms, 3))
        self.K_gradient = np.zeros((n_atoms, 3))


        # loop over all of the atoms
        for atom_index in range(n_atoms):
            for cart_index in range(3):
                deriv_index = 3 * atom_index + cart_index


                # get the two-electron integral derivatives
                eri_derivs[deriv_index] = np.asarray(self.mints.ao_tei_deriv1(atom_index)[cart_index])

                # add code to contract eri_derivs with D to get J_deriv. J_uv = 2 * sum_ls (uv|ls) D_ls 
                # note we are factoring the 2 into D here
                J_deriv[deriv_index] = oe.contract("uvls,ls->uv", eri_derivs[deriv_index, :, :, :, :], D, optimize="optimal")

                # add code to contract eri_derivs with D to get K_deriv. K_uv = -1 * sum_ls (ul|vs) D_ls
                # note we are factoring the 2 into D here
                K_deriv[deriv_index] = -1 / 2 * oe.contract("ulvs,ls->uv", eri_derivs[deriv_index, :, :, :, :], D, optimize="optimal")

                # add code to contract J_deriv with D to get J_gradient
                self.J_gradient[atom_index, cart_index] = 1 / 2 * oe.contract("uv,uv->", D, J_deriv[deriv_index, :, :], optimize="optimal")

                # add code to contract K_deriv with D to get K_gradient
                self.K_gradient[atom_index, cart_index] = 1 / 2 * oe.contract("uv,uv->", D, K_deriv[deriv_index, :, :], optimize="optimal")



    def compute_analytic_gradient(self, use_psi4=False):
        """Compute the term-by-term analytic gradient for the CQED-RHF energy.
        
        Parameters
        ----------
        use_psi4 : bool, optional
            If False (default), the method computes the gradient using internal methods.
            If True, the method uses Psi4’s built-in SCF gradient and adds CQED-specific
            corrections (dipole-dipole and quadrupole gradients).
        
        Notes
        -----
        Timing for each computational step is printed to standard output.
        """
        start_total = time.time()

        if use_psi4:
            t0 = time.time()
            self.compute_scf_gradient(qed_wfn=True)
            print(f"Time for SCF gradient via Psi4: {time.time() - t0:.3e} s")

            t1 = time.time()
            self.compute_quadrupole_gradient()
            print(f"Time for Quadrupole gradient: {time.time() - t1:.3e} s")

            t2 = time.time()
            self.compute_dipole_dipole_gradient()
            print(f"Time Dipole-dipole gradient: {time.time() - t2:.3e} s")

            self.qed_rhf_gradient = self.scf_grad + self.o_dse_gradient + self.K_dse_gradient
            print(f"Final gradient assembly using psi4: {time.time() - t0:.3e} s")

        else:
            t0 = time.time()
            self.compute_fock_matrix_term()
            print(f"Time for Fock matrix term: {time.time() - t0:.3e} s")

            t1 = time.time()
            self.compute_one_electron_integral_gradient_terms()
            print(f"Time for One-electron gradient terms: {time.time() - t1:.3e} s")

            t2 = time.time()
            self.compute_two_electron_integral_gradient_terms_2()
            print(f"Time for Two-electron gradient terms: {time.time() - t2:.3e} s")

            t3 = time.time()
            self.compute_nuclear_repulsion_gradient()
            print(f"Time for Nuclear repulsion gradient: {time.time() - t3:.3e} s")

            t4 = time.time()
            self.compute_dipole_dipole_gradient()
            print(f"Time for Dipole-dipole gradient: {time.time() - t4:.3e} s")

            t5 = time.time()
            self.compute_quadrupole_gradient()
            print(f"Time for Quadrupole gradient: {time.time() - t5:.3e} s")
            print(f"Final gradient assembly using class: {time.time() - t0:.3e} s")

            # Final gradient assembly

            self.canonical_scf_gradient = (
                self.overlap_gradient +
                self.kinetic_gradient +
                self.potential_gradient +
                self.J_gradient +
                self.K_gradient +
                self.nuclear_repulsion_gradient
            )
            self.qedrhf_gradient = self.canonical_scf_gradient + self.K_dse_gradient + self.o_dse_gradient





    def gradient_summary(self):
        # time each step
        time_start = time.time()
        self.compute_analytic_gradient(use_psi4=False)
        time_end = time.time()
        #print(f"Time to compute analytic CQED-RHF gradient: {time_end - time_start:.4f} seconds\n")
        time_start = time.time()
        self.compute_numerical_gradient()
        time_end = time.time()
        print(f"Time to compute numerical CQED-RHF gradient: {time_end - time_start:.4f} seconds\n")
        print(f"Analytical CQED-RHF Gradient:\n")
        print(self.qedrhf_gradient)
        print(f"\nNumerical CQED-RHF Gradient:\n")
        print(self.numerical_energy_gradient)
        difference = self.qedrhf_gradient - self.numerical_energy_gradient
        diff_norm = np.linalg.norm(difference)
        print(f"\nDifference between analytical and numerical gradient: {diff_norm:.4e}\n")
        print("Using Psi4 to compute the canonical RHF gradient for comparison...\n")
        time_start = time.time()
        self.compute_analytic_gradient(use_psi4=True)
        time_end = time.time()
        #print(f"Time to compute analytic CQED-RHF gradient using Psi4: {time_end - time_start:.4f} seconds\n")
        print(F"Analytical Canonical Gradient:\n")
        print(self.canonical_scf_gradient)
        print(f"Analytical Canonical Gradient using Psi4:\n")
        print(self.scf_grad)
        difference = self.canonical_scf_gradient - self.scf_grad
        diff_norm = np.linalg.norm(difference)
        print(f"\nDifference between analytical and Psi4 canonical gradient: {diff_norm:.4e}\n")


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
