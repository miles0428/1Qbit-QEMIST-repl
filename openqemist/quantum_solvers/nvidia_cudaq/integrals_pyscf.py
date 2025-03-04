#   Copyright 2019 1QBit
#   
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
from pyscf import ao2mo,lib

def compute_integrals_fragment(mol, myhf):
    """ 
        Compute the electronic integrals and store them in a data-structure that can be used
        by MicrosoftQSharp backend
    """
    
    from pyscf import gto, scf, fci, ao2mo

    n_orbitals = len(myhf.mo_energy)
    
    # One electron integrals: compute with PySCF, then thresholds and prepare for Microsoft data-structure
    one_electron_compressed = myhf.mo_coeff.T @ myhf.get_hcore() @ myhf.mo_coeff
    t1 = one_electron_compressed.reshape(n_orbitals, n_orbitals).astype(float)
    t1_thresh = np.around(t1, decimals=6)
    t1_bb = [([int(j+1), int(i+1)], t1_thresh[i, j])
             for i in range(t1_thresh.shape[0]) for j in range(i, t1_thresh.shape[1])
             if t1_thresh[i, j] != 0.]

    # Two electron integrals: compute with PySCF, then thresholds and prepare for Microsoft data-structure

    # The following occasionally returned a PySCF bug. It was replaced by the snippet below
    #two_electron_compressed = ao2mo.kernel(mol, myhf.mo_coeff)
    #t2 = ao2mo.restore(1, two_electron_compressed, n_orbitals)

    twoint=myhf._eri
    eri= ao2mo.restore(8,twoint,n_orbitals)
    eri= ao2mo.incore.full(eri,myhf.mo_coeff)
    t2 = ao2mo.restore(1,eri,n_orbitals)

    # Store entries for non-zero integrals only if they are not a permutation
    # of an existing entry
    
    def is_two_body_permutation(coords1, coords2):
        """ Test if coords are two-electorn permutations in mulliken convention """
        return (
            coords1 == [coords2[i] for i in [0, 1, 2, 3]] or
            coords1 == [coords2[i] for i in [2, 3, 0, 1]] or
            coords1 == [coords2[i] for i in [1, 0, 3, 2]] or
            coords1 == [coords2[i] for i in [3, 2, 1, 0]] or
            coords1 == [coords2[i] for i in [1, 0, 2, 3]] or
            coords1 == [coords2[i] for i in [3, 2, 0, 1]] or
            coords1 == [coords2[i] for i in [0, 1, 3, 2]] or
            coords1 == [coords2[i] for i in [2, 3, 1, 0]])

    t2_thresh = np.around(t2, decimals=6)
    values, indices = [], []

    for (coords, value) in np.ndenumerate(t2_thresh):
        if value == 0.: continue
        else :
            coords = list(coords)
            new_integral = True
            for coords_ref in indices:
                if is_two_body_permutation(coords_ref, coords):
                    new_integral = False
            if new_integral:
                values.append(value)
                indices.append(coords)

    # Convert two-electron integrals to Broombridge format
    t2_bb = [[int(coord+1) for coord in coords]  for coords in indices]
    t2_bb = [(t2_bb[i], values[i]) for i in range(len(t2_bb))]

    return t1_bb, t2_bb

def active_space_unpolarized(mf, ncore=0, nact=None):
        einsum = lib.einsum

        if nact is None:
            nact = mf.mo_coeff.shape[1] - ncore

        nuclear = mf.mol.energy_nuc()
        ecore = 0.0

        mo_coeff_core = mf.mo_coeff[:, :ncore]
        mo_coeff_acti = mf.mo_coeff[:, ncore : ncore + nact]

        h1atom = mf.get_hcore()
        if ncore != 0:
            dm_ao_core = 2 * mo_coeff_core @ mo_coeff_core.T
            vj, vk = mf.get_jk(mf.mol, dm_ao_core)
            veff_ao = vj - 0.5 * vk
            ecore += einsum("ij,ji->", dm_ao_core, h1atom + 0.5 * veff_ao)
            h1atom += veff_ao
        hpq = einsum("ap,bq,ab->pq", mo_coeff_acti, mo_coeff_acti, h1atom)
        hpqrs = ao2mo.full(mf.mol, mo_coeff_acti, compact=False).reshape(
            nact, nact, nact, nact
        )
        hpqrs = hpqrs.transpose(0, 2, 3, 1)

        act_nelec = mf.mol.nelectron - ncore * 2

        return nact, act_nelec, nuclear + ecore, hpq, hpqrs
