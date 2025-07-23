import psi4
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from oo_cqed_rhf import CQEDRHFCalculator


def coulomb_matrix(geom: np.ndarray,
                   charges: np.ndarray,
                   lambda_vec: np.ndarray,
                   alpha: float = 50.0) -> np.ndarray:
    """
    Build the Coulomb matrix augmented by the cavity term for one geometry.

    Args:
        geom        : (n_atoms, 3) array of Cartesian coords
        charges     : (n_atoms,) array of nuclear charges Z_i
        lambda_vec  : (3,)  cavity polarization vector λ
        alpha       : scaling factor for cavity term

    Returns:
        cm          : (n_atoms, n_atoms) Coulomb matrix
    """
    n = charges.size
    cm = np.zeros((n, n))
    # off‐diagonals
    for i in range(n):
        for j in range(i+1, n):
            Rij = np.linalg.norm(geom[i] - geom[j])
            val = charges[i] * charges[j] / Rij
            cm[i, j] = cm[j, i] = val

    # diagonals
    for i in range(n):
        Zi = charges[i]
        dot = geom[i].dot(lambda_vec)
        cm[i, i] = 0.5 * Zi**2.4 + alpha * (Zi * dot)**2

    return cm

def build_dataset(mol_geoms: list[np.ndarray],
                  mol_charges: list[np.ndarray],
                  lambdas:   list[np.ndarray],
                  energies:  list[float],
                  flatten:   bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn lists of (geom, charges, λ) and energies into X, y arrays.
    """
    X = []
    for geom, q, lam in zip(mol_geoms, mol_charges, lambdas):
        cm = coulomb_matrix(geom, q, lam)
        X.append(cm.flatten() if flatten else cm)
    return np.vstack(X), np.array(energies)

def split_dataset(X: np.ndarray,
                  y: np.ndarray,
                  train_size: float = 0.2,
                  random_state: int = 42):
    """
    Convenience wrapper around sklearn.train_test_split.
    """
    return train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state
    )


# ----- 1) Generate your collection of geoms, charges, lambdas, energies -----
def generate_h2o_stretch(rs: np.ndarray,
                         lambda_vec: np.ndarray,
                         psi4_opts: dict) -> tuple:
    geoms, qs, ls, Es = [], [], [], []
    for r in rs:
        # build molecule
        mol_str = f"""
        O
        H 1 {r}
        H 1 {r} 2 104.5
        symmetry c1
        """
        mol = psi4.geometry(mol_str)
        geoms.append(mol.geometry().to_array())
        qs.append(np.array([mol.fZ(i) for i in range(3)]))
        ls.append(lambda_vec.copy())

        # compute energy
        calc = CQEDRHFCalculator(lambda_vec, mol_str, psi4_opts)
        calc.calc_cqed_rhf_energy()
        Es.append(calc.cqed_rhf_energy)
    return geoms, qs, ls, Es

# parameters
rs = np.linspace(0.5, 1.2, 31)
lambda_vecs = [np.array([0.0, 0.0, lz])  for lz in np.linspace(0.0, 0.1, 31)]
# (you can mix-and-match: e.g. Cartesian grid of angles & magnitudes)

psi4_options = {
    "basis": "cc-pVDZ",
    "save_jk": True,
    "scf_type": "pk",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}
psi4.set_options(psi4_options)

geoms, qs, lambdas, Es = generate_h2o_stretch(rs, np.array([0,0,0.05]), psi4_options)

# ----- 2) Build and split dataset -----
X, y = build_dataset(geoms, qs, lambdas, Es)

# create an index array 0,1,2,…,30
idx = np.arange(len(rs))

# split X, y AND idx all together
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, idx,
    train_size=0.2,
    random_state=1,
    shuffle=True
)

# now idx_train / idx_test tell you which rows went where
print("TRAIN bond lengths:", rs[idx_train])
print(" TEST bond lengths:", rs[idx_test])

# ----- 3) Train KRR with grid-search CV -----
param_grid = {
    'alpha': np.logspace(-12, 12, num=12),
    'gamma': np.logspace(-12, 12, num=12)
}

krr = GridSearchCV(KernelRidge(kernel='rbf'),
                   param_grid,
                   cv=4,
                   scoring='neg_mean_squared_error')
krr.fit(X_train, y_train)
y_pred = krr.predict(X_test)

# ----- 4) Plot -----
plt.plot(rs, Es, label='True PES')
# note: X_test corresponds to a subset of rs
test_rs = np.array(rs)[np.isin(range(len(rs)), sorted(range(len(rs)))[:len(y_test)], invert=True)]
plt.plot(rs[idx_test], y_pred, 'o', label='KRR Predictions')
plt.xlabel('r / Å')
plt.ylabel('Energy / E_h')
plt.legend()
plt.show()

