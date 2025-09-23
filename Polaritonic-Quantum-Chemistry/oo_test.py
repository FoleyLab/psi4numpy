from oo_cqed_rhf import CQEDRHFCalculator
import numpy as np
import psi4

## forward displaced geometry string
h2o_string = """
         O            0.000000000000     0.000000000000    -0.068000000000
         H            0.000000000000    -0.790689573744     0.543701060715
         H            0.000000000000     0.790689573744     0.543701060715
no_reorient
nocom
symmetry c1
"""


# lambda vector along z
lambda_vector = np.array([0, 0, 0.05])


# psi4 options
psi4_options = {
    "basis": "cc-pVDZ",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

# instantiate, using forward displaced geometry string... shouldn't matter because
# we will pass our desired geometry string when we want to compute the gradient
calc = CQEDRHFCalculator(lambda_vector, h2o_string, psi4_options)

print("RUNNING WITH PSI4 SCF_GRAD=TRUE")
# calculate the CQED-RHF energy and gradient at h2o_string_b, use our routines for all terms
qed_rhf_energy, qed_rhf_grad_df, g = calc.calc_force_and_energy(h2o_string, use_psi4_scf_grad=True)

print("RUNNING WITH PSI4 SCF_GRAD=FALSE")
# calculate the CQED-RHF energy and gradient at h2o_string_b, use our routines for all terms
qed_rhf_energy, qed_rhf_grad_pk, g = calc.calc_force_and_energy(h2o_string, use_psi4_scf_grad=False)

diff = np.linalg.norm(qed_rhf_grad_df - qed_rhf_grad_pk)
print(F"Diff is {diff:12.10e}")
