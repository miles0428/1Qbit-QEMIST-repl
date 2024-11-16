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

import itertools
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.mappers import JordanWignerMapper

def make_ucc_ansatz(n_spinorbitals, n_elec):
    """
    make_ucc_ansatz
    This function creates a UCCSD ansatz for a given molecule and basis set
    Args:
        n_spinorbitals(int): integer representing the number of spinorbitals (qubits) for
                             a given molecule and basis set
        n_elec(int):         integer representing the number of electrons for a given molecule
    Returns:
        ucc(UCC):            UCC object representing the UCCSD ansatz for a given molecule
    """
    n_spin_orb = n_spinorbitals
    n_elec_per_spin = n_elec // 2
    n_spatial_orb = n_spin_orb // 2

    # first create the HF state
    n_qubits = n_spin_orb
    hf = QuantumCircuit(n_qubits)
    for i in range(n_elec):
        hf.x(i)

    # num_particles: we need to separate number of spin up and spin down electrons
    # excitations: sd -> singles + doubles
    # qubit_mapper: The Jordan Wigner mapper
    # initial state is HF
    uccsd = UCCSD(
        num_spatial_orbitals=n_spatial_orb,
        num_particles=(n_elec_per_spin,n_elec- n_elec_per_spin),
        reps = 1,
        qubit_mapper=JordanWignerMapper(),
        initial_state=hf,
    )

    return uccsd