"""
Molecular Dynamics Simulation using QED-RHF
===========================================
This script performs a velocity Verlet molecular dynamics simulation
for nitrobenzene using cavity QED-RHF calculations.
"""

import sys
from typing import TextIO

import numpy as np
import psi4

from oo_cqed_rhf import CQEDRHFCalculator

# Suppress Psi4 output
psi4.core.be_quiet()


# =============================================================================
# Constants (Atomic Units)
# =============================================================================
BOHR_TO_ANGSTROM = 0.529177210903
FS_TO_AU = 41.34137314
AMU_TO_AU = 1822.8884850


# =============================================================================
# Geometry Conversion Functions
# =============================================================================
def geom_bohr_to_angstrom_string(geom_bohr: np.ndarray,
                                 symbols: list[str]) -> str:
    """
    Convert geometry from Bohr to Angstrom and format for Psi4.

    Parameters
    ----------
    geom_bohr : np.ndarray
        (N, 3) array of atomic coordinates in Bohr
    symbols : list[str]
        List of atomic symbols

    Returns
    -------
    str
        Psi4-formatted geometry string in Angstrom with directives
    """
    geom_ang = geom_bohr * BOHR_TO_ANGSTROM

    lines = []
    for sym, (x, y, z) in zip(symbols, geom_ang):
        lines.append(f"{sym:2s}  {x:14.12f}  {y:14.12f}  {z:14.12f}")

    # Append Psi4 directives
    suffix = """1 1
no_reorient
nocom
symmetry c1"""

    return "\n".join(lines + [suffix])


# =============================================================================
# Molecular Dynamics Functions
# =============================================================================
def velocity_verlet_step(x_bohr: np.ndarray,
                         v_bohr: np.ndarray,
                         grad: np.ndarray,
                         masses: np.ndarray,
                         dt: float,
                         calc: CQEDRHFCalculator,
                         symbols: list[str],
                         frame: int,
                         use_psi4_scf_grad: bool = False,
                         out: TextIO = sys.stdout) -> tuple:
    """
    Perform one Velocity-Verlet MD step and write XYZ frame.

    Parameters
    ----------
    x_bohr : np.ndarray
        (N, 3) positions in Bohr
    v_bohr : np.ndarray
        (N, 3) velocities in Bohr/atomic-time
    grad : np.ndarray
        (N, 3) gradient in Hartree/Bohr
    masses : np.ndarray
        (N,) masses in atomic units
    dt : float
        Time step in atomic units
    calc : CQEDRHFCalculator
        Calculator object for energy and forces
    symbols : list[str]
        List of atomic symbols
    frame : int
        Frame number for trajectory
    use_psi4_scf_grad : bool, optional
        Whether to use Psi4's SCF gradient
    out : TextIO, optional
        Output file for XYZ trajectory

    Returns
    -------
    tuple
        (x_new, v_new, E_new, grad_new, g_new)
    """
    n_atoms = x_bohr.shape[0]

    # Current forces and accelerations
    F_old = -grad
    a_old = F_old / masses[:, None]

    # Update positions
    x_new = x_bohr + v_bohr * dt + 0.5 * a_old * dt**2

    # Calculate new forces
    geom_block_new = geom_bohr_to_angstrom_string(x_new, symbols)
    E_new, grad_new, g_new = calc.calc_force_and_energy(
        geom_block_new, use_psi4_scf_grad=use_psi4_scf_grad
    )
    F_new = -grad_new
    a_new = F_new / masses[:, None]

    # Update velocities
    v_new = v_bohr + 0.5 * (a_old + a_new) * dt

    # Write XYZ frame
    _write_xyz_frame(n_atoms, frame, E_new, symbols, x_new, out)

    return x_new, v_new, E_new, grad_new, g_new


def _write_xyz_frame(n_atoms: int, frame: int, energy: float,
                     symbols: list[str], coords_bohr: np.ndarray,
                     out: TextIO):
    """Write a single XYZ frame to output file."""
    print(n_atoms, file=out)
    print(f"Frame {frame}   E = {energy:.8f} Ha", file=out)
    
    coords_ang = coords_bohr * BOHR_TO_ANGSTROM
    for sym, (x, y, z) in zip(symbols, coords_ang):
        print(f"{sym:<2} {x: .6f} {y: .6f} {z: .6f}", file=out)


# =============================================================================
# Main Simulation
# =============================================================================
def main():
    """Run the molecular dynamics simulation."""
    
    # Cavity field parameters (lambda vector along z)
    #lambda_vector = np.array([0.0, 0.05, 0.05])
    lambda_vector = np.array([0.7839420737139418, 0.5571186332860504, 0.27395921869243256]) * 0.1
    # this is the theta = 74.1 and phi = 35.0, scaled by 0.1 atomic units

    # Psi4 calculation options
    psi4_options = {
        "basis": "6-31G",
        "save_jk": True,
        "scf_type": "pk",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }
    psi4.set_options(psi4_options)

    # Initial geometry (nitrobenzene)
    mol_string = """
 C                  0.51932475    1.23303451   -0.03194925
 C                  1.94454413    1.26916358   -0.03672882
 C                  2.62037793    0.09283428   -0.02499003
 C                 -0.19603352    0.03013062    0.00102732
 H                 -0.02069420    2.17423764   -0.04336646
 H                  2.48281698    2.20891057   -0.03611879
 H                 -1.27770137    0.03990295    0.01166953
 N                  4.09213475    0.09594076    0.03662979
 O                  4.63930696   -1.02169275    0.14459220
 O                  4.66489883    1.19839699   -0.02327545
 C                  0.49428518   -1.16712649    0.02099746
 H                 -0.03251071   -2.11492669    0.05447935
 C                  1.96291176   -1.21653219   -0.02111314
 H                  2.44359113   -1.96306433    0.61513886
 Br                 2.17304025   -1.94912156   -1.90618750
 1 1
 units angstrom
 no_com
 no_reorient
"""

    # Initialize Psi4 molecule and get initial energy
    mol = psi4.geometry(mol_string)
    e_rhf = psi4.energy("scf")
    print(f"RHF energy: {e_rhf:.12f} Ha")

    # Extract atomic information
    natoms = mol.natom()
    atom_mass = np.array([mol.mass(i) for i in range(natoms)]) * AMU_TO_AU
    symbols = [mol.symbol(i) for i in range(natoms)]
    x0_bohr = mol.geometry().to_array()

    print(f"\nNumber of atoms: {natoms}")
    print(f"Atomic masses (au):\n{atom_mass}\n")

    # Initialize QED-RHF calculator
    calc = CQEDRHFCalculator(lambda_vector, mol_string, psi4_options)

    # Calculate initial QED-RHF energy and gradient
    qed_rhf_energy, qed_rhf_grad, qed_rhf_g = calc.calc_force_and_energy(
        mol_string, use_psi4_scf_grad=True
    )
    print(f"QED-RHF initial energy: {qed_rhf_energy:.12f} Ha")
    print(f"QED-RHF initial gradient:\n{qed_rhf_grad}\n")

    # MD simulation parameters
    dt = 15.0  # time step in atomic units
    n_steps = 10
    output_file = "nitrobenzene_traj_dt_15_lx_0.05_lz_0.05_no_df.xyz"

    # Initialize MD variables
    x_curr = np.copy(x0_bohr)
    v_curr = np.zeros_like(x_curr)
    grad_curr = qed_rhf_grad

    # Run MD simulation
    print(f"Starting MD simulation ({n_steps} steps, dt = {dt} au)")
    print(f"Output trajectory: {output_file}\n")

    energy_list = []
    coupling_list = []

    with open(output_file, "w") as traj:
        for i in range(n_steps):
            x_curr, v_curr, E, grad_curr, g_new = velocity_verlet_step(
                x_curr, v_curr, grad_curr, atom_mass, dt,
                calc, symbols, frame=i,
                use_psi4_scf_grad=True,
                out=traj
            )
            energy_list.append(E)
            coupling_list.append(g_new)
            print(f"Step {i+1}/{n_steps}: E = {E:.8f} Ha, g = {g_new:.6f}")

    print(f"\nSimulation complete. Trajectory saved to {output_file}")


if __name__ == "__main__":
    main()