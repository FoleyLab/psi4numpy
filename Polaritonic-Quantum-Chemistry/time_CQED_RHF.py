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
from helper_CQED_RHF import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory("2 GB")
psi4.core.set_output_file("output.dat", False)

numpy_memory = 2


# options for H2O
h2o_options_dict = {
    "basis": "cc-pVDZ",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}


# molecule string for H2O
h2o_string = """

0 1
C -1.219262838364 -0.691927313805 0.000010980109
C -1.208864450455 0.709970533848 0.000027145785
C -0.010381917469 -1.401882171631 -0.000000820648
C 0.010416744277 1.401915431023 0.000003339373
C 1.208898305893 -0.709937453270 -0.000004130281
C 1.219296813011 0.691960453987 0.000004793658
H -2.162761211395 -1.227365016937 0.000034040277
H -2.144315481186 1.259346365929 0.000002431093
H -0.018430225551 -2.486694812775 -0.000036539052
H 0.018459986895 2.486727952957 0.000057592049
H 2.144349575043 -1.259313106537 0.000016176735
H 2.162794828415 1.227399110794 -0.000015010323
no_reorient
symmetry c1
"""

# energy for H2O from hilbert package described in [DePrince:2021:094112]
expected_h2o_e = -76.016355284146

# electric field for H2O - polarized along z-axis with mangitude 0.05 atomic units
lam_h2o = np.array([0.0, 0.0, 0.05])


# run cqed_rhf on H2O
import time
start_t = time.time()
h2o_dict = cqed_rhf(lam_h2o, h2o_string, h2o_options_dict)
end_t = time.time()
print("Time required is ", end_t-start_t)

