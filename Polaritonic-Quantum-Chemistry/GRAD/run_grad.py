
from helper_rhf_grad import RHFGrad
import numpy as np


molecule_string = """
0 1
O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587
symmetry c1
"""


# Define Psi4 options
psi4_options_dict = {
    "basis": "sto-3g",
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12
}

# Define lambda vector (example values)
lambda_vector = [0.1, 0.2, 0.3]

# Create options dictionary
options = {
    "molecule_string": molecule_string,
    "psi4_options_dict": psi4_options_dict,
    "lambda_vector": lambda_vector
}

# Instantiate the RHFGrad class
rhf_grad = RHFGrad(options)

# Compute the RHF wavefunction
rhf_wfn = rhf_grad.compute_rhf_wfn()
print(f"RHF Energy: {rhf_grad.rhf_energy}")

# Compute the gradient quantities
gradient = rhf_grad.compute_gradient_quantities()

expected_gradient = np.array(
 [[ 0.,                 0.000000000000001,  0.061008877848977],
 [-0.,                -0.023592244369028, -0.030504438924483],
 [ 0.,                 0.023592244369025, -0.030504438924483]]
)



_expected_nuclear_gradient = np.array([
    [0.00000000000000,  0.00000000000000,  2.99204046891092],
    [0.00000000000000, -2.05144597283373, -1.49602023445546],
    [0.00000000000000,  2.05144597283373, -1.49602023445546],
])

_overlap_gradient = np.array([
    [-0.00000000000000, -0.00000000000000,  0.30728746121587],
    [ 0.00000000000000, -0.14977126575800, -0.15364373060793],
    [-0.00000000000000,  0.14977126575800, -0.15364373060793],
])

_potential_gradient = np.array([
    [-0.00000000000000,  0.00000000000002, -6.81982772856799],
    [-0.00000000000000,  4.38321774316664,  3.40991386428399],
    [ 0.00000000000000, -4.38321774316666,  3.40991386428400],
])
_kinetic_gradient = np.array([
    [ 0.00000000000000, -0.00000000000000,  0.66968290617933],
    [ 0.00000000000000, -0.43735698924315, -0.33484145308966],
    [-0.00000000000000,  0.43735698924315, -0.33484145308967],
])

_coulomb_gradient = np.array([
    [ 0.00000000000000, -0.00000000000002,  3.34742251141627],
    [ 0.00000000000000, -2.03756324433539, -1.67371125570813],
    [-0.00000000000000,  2.03756324433541, -1.67371125570814],
])

_exchange_gradient = np.array([
    [-0.00000000000000,  0.00000000000000, -0.43559674130726],
    [-0.00000000000000,  0.26932748463493,  0.21779837065363],
    [ 0.00000000000000, -0.26932748463493,  0.21779837065363],
])

# Check the computed gradient against the expected gradient
print("Checking nuclear gradient")
print(np.allclose(rhf_grad.nuclear_energy_gradient.reshape(3,3), _expected_nuclear_gradient, atol=1e-6))

print("Checking kinetic gradient")
print(np.allclose(rhf_grad.kinetic_gradient.reshape(3,3), _kinetic_gradient, atol=1e-6))
# print both kinetic_gradient and _kinetic_gradient
print("Calculated kinetic gradient:")
print(rhf_grad.kinetic_gradient.reshape(3,3))
print("Expected kinetic gradient:")
print(_kinetic_gradient)

print("Calculated potential gradient:")
print(rhf_grad.potential_gradient.reshape(3,3))
print("Expected potential gradient:")
print(_potential_gradient)

print("Checking potential gradient")
print(np.allclose(rhf_grad.potential_gradient.reshape(3,3), _potential_gradient, atol=1e-6))

print("Checking overlap gradient")
print(np.allclose(rhf_grad.pulay_force.reshape(3,3), _overlap_gradient, atol=1e-6))

print("Checking J gradient")
print(np.allclose(rhf_grad.J_gradient.reshape(3,3), _coulomb_gradient, atol=1e-6))
print("Calculated J gradient:")
print(rhf_grad.J_gradient.reshape(3,3))
print("Expected J gradient:")
print(_coulomb_gradient)
print("Checking K gradient")
print(np.allclose(rhf_grad.K_gradient.reshape(3,3), _exchange_gradient, atol=1e-6))
print("Calculated K gradient:")
print(rhf_grad.K_gradient.reshape(3,3))
print("Expected K gradient:")
print(_exchange_gradient)


print(f"Computed Gradient:\n {gradient.reshape(3,3)}")
print(f"Expected Gradient:\n {expected_gradient}")
print("Checking total gradient")
print(np.allclose(gradient.reshape(3,3), expected_gradient, atol=1e-6))
