#%%                       IMPORTATION OF LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from solrad.atmosphere.angstrom_exponent import *

#%%                           OTHER TESTS

# Let us check the variation of the Angstrom exponent with respected to
# relative humidity, for both spectral ranges (below and above 500nm), for
# all models.
RHs = np.linspace(0, 100, 101)

for model in ["Rural", "Urban", "Maritime"]:

    fig = plt.figure(figsize=(15,10))
    alpha1 = compute_angstrom_exponent_using_SF(RHs, 499, model=model)
    alpha2 = compute_angstrom_exponent_using_SF(RHs, 500, model=model)
    plt.plot(RHs, alpha1, label="λ < 500 nm")
    plt.plot(RHs, alpha2, label="λ ≥ 500 nm")
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel("RH [%]")
    plt.ylabel("Angstrom Exponent")
    plt.title(f"Model = {model}")
    plt.legend(title = "Spectral Range")
    plt.show()

# %%
