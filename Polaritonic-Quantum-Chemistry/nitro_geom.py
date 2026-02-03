"""
Geometry Optimization using BFGS and QED-RHF
============================================
This script performs a BFGS quasi-Newton geometry optimization
for nitrobenzene using cavity QED-RHF calculations.
"""

import numpy as np
import psi4

from oo_cqed_rhf import CQEDRHFCalculator

# Suppress Psi4 output
psi4.core.be_quiet()


# =============================================================================
# Constants (Atomic Units)
# =============================================================================
BOHR_TO_ANGSTROM = 0.529177210903
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
# BFGS Optimization Functions
# =============================================================================
def bfgs_update(xk: np.ndarray, gk: np.ndarray,
                xkp1: np.ndarray, gkp1: np.ndarray,
                Hk: np.ndarray) -> np.ndarray:
    """
    Perform BFGS update of the Hessian approximation.

    Parameters
    ----------
    xk : np.ndarray
        Previous solution vector (flattened coordinates)
    gk : np.ndarray
        Gradient at previous solution
    xkp1 : np.ndarray
        Current solution vector
    gkp1 : np.ndarray
        Gradient at current solution
    Hk : np.ndarray
        Current Hessian approximation

    Returns
    -------
    np.ndarray
        Updated Hessian approximation
    """
    # Compute yk (gradient difference)
    yk = np.matrix(gkp1 - gk).T
    
    # Compute sk (position difference)
    sk = np.matrix(xkp1 - xk).T
    
    # Compute alpha denominator for Hessian update
    a_d = yk.T @ sk
    alpha = 1 / a_d[0, 0]
    
    # Compute beta denominator for Hessian update
    b_d = sk.T @ Hk @ sk
    beta = -1 / b_d[0, 0]
    
    # Check for numerical instability
    if np.isclose(alpha, 0):
        print("Warning: y·s ≈ 0. Skipping BFGS update.")
        return Hk
    if np.isclose(beta, 0):
        print("Warning: s·H·s ≈ 0. Skipping BFGS update.")
        return Hk
    
    # BFGS update formula: H_{k+1} = H_k + α·y·y^T - β·H·s·s^T·H^T
    B1 = alpha * yk @ yk.T
    B2 = beta * Hk @ sk @ sk.T @ Hk.T
    Hkp1 = Hk + B1 + B2
    
    return Hkp1


def optimize_geometry(calc: CQEDRHFCalculator,
                      x0_bohr: np.ndarray,
                      symbols: list[str],
                      bfgs_update_func,
                      tol: float = 1e-6,
                      max_iter: int = 50,
                      save_trajectory: bool = True,
                      traj_file: str = "optimization.xyz") -> tuple:
    """
    BFGS geometry optimizer using QED-RHF forces.

    Parameters
    ----------
    calc : CQEDRHFCalculator
        Calculator for energy and forces
    x0_bohr : np.ndarray
        Initial geometry (N_atoms, 3) in Bohr
    symbols : list[str]
        List of atomic symbols
    bfgs_update_func : callable
        BFGS Hessian update function
    tol : float, optional
        Convergence tolerance for gradient norm
    max_iter : int, optional
        Maximum number of iterations
    save_trajectory : bool, optional
        Whether to save optimization trajectory
    traj_file : str, optional
        Trajectory file name

    Returns
    -------
    tuple
        (optimized_geometry, final_energy, final_gradient, final_hessian)
    """
    n_atoms = x0_bohr.shape[0]
    ndim = 3 * n_atoms
    
    # Flatten initial geometry
    xk = x0_bohr.reshape(ndim)
    
    # Initial Hessian approximation (identity matrix)
    Hk = np.eye(ndim)
    
    # Calculate initial energy and gradient
    geom_block = geom_bohr_to_angstrom_string(xk.reshape(n_atoms, 3), symbols)
    Ek, gk_full, g = calc.calc_force_and_energy(
        geom_block, use_psi4_scf_grad=True
    )
    gk = gk_full.reshape(ndim)
    
    # Open trajectory file if requested
    traj = None
    if save_trajectory:
        traj = open(traj_file, "w")
        _write_xyz_frame(traj, symbols, xk.reshape(n_atoms, 3), Ek, 0)
    
    print("="*70)
    print("Starting BFGS Geometry Optimization")
    print("="*70)
    print(f"Initial energy: {Ek:.8f} Eh")
    print(f"Initial ‖g‖:    {np.linalg.norm(gk):.2e}")
    print(f"Convergence tolerance: {tol:.2e}")
    print(f"Maximum iterations: {max_iter}")
    print("-"*70)
    
    converged = False
    for iteration in range(1, max_iter + 1):
        # 1) Compute search direction: pk = -Hk^{-1} · gk
        pk = -np.linalg.solve(Hk, gk)
        
        # 2) Update geometry
        xkp1 = xk + pk
        
        # 3) Evaluate energy and gradient at new geometry
        geom_block = geom_bohr_to_angstrom_string(
            xkp1.reshape(n_atoms, 3), symbols
        )
        Ekp1, gkp1_full, g = calc.calc_force_and_energy(
            geom_block, use_psi4_scf_grad=True
        )
        gkp1 = gkp1_full.reshape(ndim)
        
        # 4) Check convergence
        norm_g = np.linalg.norm(gkp1)
        energy_change = Ekp1 - Ek
        step_size = np.linalg.norm(pk)
        
        print(f"Iter {iteration:3d}: E = {Ekp1:14.8f} Eh   "
              f"ΔE = {energy_change:+.2e}   "
              f"‖g‖ = {norm_g:.2e}   "
              f"‖step‖ = {step_size:.2e}")
        
        # Save trajectory frame
        if save_trajectory:
            _write_xyz_frame(traj, symbols, xkp1.reshape(n_atoms, 3), 
                           Ekp1, iteration)
        
        # Check convergence
        if norm_g < tol:
            converged = True
            break
        
        # 5) BFGS update of Hessian
        Hkp1 = bfgs_update_func(xk, gk, xkp1, gkp1, Hk)
        
        # 6) Update for next iteration
        xk, gk, Hk, Ek = xkp1, gkp1, Hkp1, Ekp1
    
    # Close trajectory file
    if save_trajectory:
        traj.close()
    
    # Print final results
    print("="*70)
    if converged:
        print("✓ OPTIMIZATION CONVERGED")
    else:
        print("✗ WARNING: Maximum iterations reached without convergence")
    print("="*70)
    
    final_geom_bohr = xk.reshape(n_atoms, 3)
    final_geom_string = geom_bohr_to_angstrom_string(final_geom_bohr, symbols)
    
    print(f"\nFinal energy: {Ekp1:.10f} Eh")
    print(f"Final ‖g‖:    {norm_g:.2e}")
    print(f"\nFinal geometry (Bohr):")
    print(final_geom_bohr)
    print(f"\nFinal gradient (Eh/Bohr):")
    print(gkp1_full)
    print(f"\n{'='*70}")
    print("Final Geometry String (Angstrom):")
    print("="*70)
    print(final_geom_string)
    print("="*70)
    
    if save_trajectory:
        print(f"\nTrajectory saved to: {traj_file}")
    
    return final_geom_bohr, Ekp1, gkp1_full, Hk


def _write_xyz_frame(file_handle, symbols: list[str], coords_bohr: np.ndarray,
                    energy: float, iteration: int):
    """Write a single XYZ frame to file."""
    n_atoms = len(symbols)
    coords_ang = coords_bohr * BOHR_TO_ANGSTROM
    
    print(n_atoms, file=file_handle)
    print(f"Iteration {iteration}   E = {energy:.8f} Ha", file=file_handle)
    for sym, (x, y, z) in zip(symbols, coords_ang):
        print(f"{sym:<2} {x: .6f} {y: .6f} {z: .6f}", file=file_handle)


# =============================================================================
# Main Optimization
# =============================================================================
def main():
    """Run the geometry optimization."""
    
    # Cavity field parameters (lambda vector)
    #lambda_vector = np.array([0.0, 0.05, 0.05])
    lambda_vector = np.array([0.7839420737139418, 0.5571186332860504, 0.27395921869243256]) * 0.1

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
1 1
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
 units angstrom
 no_com
 no_reorient
"""

    # Initialize Psi4 molecule and get initial energy
    mol = psi4.geometry(mol_string)
    e_rhf = psi4.energy("scf")
    print(f"\nStandard RHF energy: {e_rhf:.12f} Ha\n")

    # Extract atomic information
    natoms = mol.natom()
    symbols = [mol.symbol(i) for i in range(natoms)]
    x0_bohr = mol.geometry().to_array()

    # Initialize QED-RHF calculator
    calc = CQEDRHFCalculator(lambda_vector, mol_string, psi4_options)

    # Optimization parameters
    convergence_tol = 1e-6  # Gradient norm convergence threshold
    max_iterations = 50
    trajectory_file = "nitrobenzene_opt_lx_0.05_lz_0.05.xyz"

    # Run geometry optimization
    optimized_geom, final_energy, final_grad, final_hess = optimize_geometry(
        calc=calc,
        x0_bohr=x0_bohr,
        symbols=symbols,
        bfgs_update_func=bfgs_update,
        tol=convergence_tol,
        max_iter=max_iterations,
        save_trajectory=True,
        traj_file=trajectory_file
    )

    # Save final optimized geometry to a separate file
    output_geom_file = "nitrobenzene_optimized_geometry.xyz"
    with open(output_geom_file, "w") as f:
        _write_xyz_frame(f, symbols, optimized_geom, final_energy, 0)
    print(f"Final geometry saved to: {output_geom_file}\n")


if __name__ == "__main__":
    main()
