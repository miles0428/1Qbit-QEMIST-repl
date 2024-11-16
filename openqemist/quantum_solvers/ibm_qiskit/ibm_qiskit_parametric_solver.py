'''
This module contains the IBMQiskitParametricSolver class, which is a subclass of the ParametricQuantumSolver class.

This file follows Apache License Version 2.0.

Author: YC Chung

!!! This code is modified from the microsoft_qsharp_parametric_solver.py file in the same package. !!!

!!! If you want to modify the code, make sure you also follow the license in openqemist !!!
'''

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

from enum import Enum
from ..parametric_quantum_solver import ParametricQuantumSolver
import os
import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer.primitives import Estimator as AerEstimator
import warnings
# Import python packages for Microsoft Python interops
import qsharp
import qsharp.chemistry as qsharpchem
# Import pyscf and functions making use of it
from pyscf import gto, scf
from .integrals_pyscf import compute_integrals_fragment
from .generate_uccsd_operators import make_ucc_ansatz


class IBMQiskitParametricSolver(ParametricQuantumSolver):
    """Performs an energy estimation for a molecule with a parametric circuit.

    Performs energy estimations for a given molecule and a choice of ansatz
    circuit that is supported.

    Attributes:
        n_samples (int): The number of samples to take from the hardware emulator.
        optimized_amplitudes (list): The optimized amplitudes.
        verbose(bool): Toggles the printing of debug statements.
    """

    class Ansatze(Enum):
        """ Enumeration of the ansatz circuits that are supported."""
        UCCSD = 0

    def __init__(self, ansatz, molecule, mean_field = None):
        """Initialize the settings for simulation.

        If the mean field is not provided, it is automatically calculated.

        Args:
            ansatz (IBMQiskitParametricSolver.Ansatze): Ansatz for the quantum solver.
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF, optional): The mean field of the molecule. Defaults to None.
        """
        assert isinstance(ansatz, IBMQiskitParametricSolver.Ansatze)
        self.verbose = False

        # Initialize the number of samples to be used by the MicrosoftQSharp backend
        self.n_samples = 1e18

        # Initialize the amplitudes (parameters to be optimized)
        self.optimized_amplitudes = []

        # Obtain fragment info with PySCF
        # -----------------------------------------

        # Compute mean-field if not provided. Check that it has converged
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()

        if not mean_field.converged:
            warnings.warn("IBMQiskitParametricSolver simulating with mean field not converged.",
                          RuntimeWarning)

        # Set molecule and mean field attributes
        self.n_orbitals = len(mean_field.mo_energy)
        self.n_spin_orbitals = 2 * self.n_orbitals
        self.n_electrons = molecule.nelectron
        nuclear_repulsion = mean_field.energy_nuc()

        # Compute and set values of electronic integrals
        # ----------------------------------------------

        self.uccsd_ansatz = make_ucc_ansatz(self.n_spin_orbitals, self.n_electrons)
        n_amplitudes = self.uccsd_ansatz.num_parameters
        self.amplitude_dimension = n_amplitudes
        amplitudes = np.ones((n_amplitudes), dtype=np.float64)

        # Get data-structure to store problem description
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        filename = os.path.join(__location__, 'dummy_0.2.yaml')
        molecular_data = qsharpchem.load_broombridge(filename)

        # Compute one and two-electron integrals, store them in the Microsoft data-structure
        integrals_one, integrals_two = compute_integrals_fragment(molecule, mean_field)
        molecular_data.problem_description[0].hamiltonian['OneElectronIntegrals']['Values'] = integrals_one
        molecular_data.problem_description[0].hamiltonian['TwoElectronIntegrals']['Values'] = integrals_two
        molecular_data.problem_description[0].coulomb_repulsion['Value'] = nuclear_repulsion

        # Generate Fermionic and then JW Hamiltonians
        # ----------------------------------------------

        # C# Chemistry library: Compute fermionic Hamiltonian
        self.ferm_hamiltonian = molecular_data.problem_description[0].load_fermion_hamiltonian()

        # Qiskit Nature library: Convert Fermionic Hamiltonian to FermionicOp
        self.jw_hamiltonian = JordanWignerMapper().map(self._QsharpFermionToQiskitEncoding(self.ferm_hamiltonian))



    def _QsharpFermionToQiskitEncoding(self,ferm_hamiltonian:qsharpchem.FermionHamiltonian)->FermionicOp:
        '''
        Convert a FermionHamiltonian object from the Q# Chemistry library to a FermionicOp object from the Qiskit Nature library.

        args:
            ferm_hamiltonian (qsharp.chemistry.FermionHamiltonian): The FermionHamiltonian object from the Q# Chemistry library.
        
        returns:
            (qiskit_nature.second_q.operators.fermionic_op.FermionicOp): The FermionicOp object from the Qiskit Nature library.
        '''
        ferm_hamiltonian = ferm_hamiltonian.__dict__
        num_spin_orbital = len(ferm_hamiltonian["system_indices"])
        data = {}
        for _,FermOps in ferm_hamiltonian["terms"]:
            for elelment in FermOps:
                operator = elelment[0][0]
                coefficient1 = elelment[0][1]
                coefficient2 = elelment[1]
                operator_key = ""
                for SpinString,Qubit in operator:
                    spin = "+" if (SpinString=="u") else "-"
                    operator_key += f"{spin}_{num_spin_orbital-Qubit-1} "
                if operator_key[:-1] not in data:
                    data[operator_key[:-1]] = coefficient1*coefficient2*0.5
                else :
                    data[operator_key[:-1]] += coefficient1*coefficient2*0.5
                operator_key = ""
                for SpinString,Qubit in operator:
                    spin = "-" if (SpinString=="u") else "+"
                    operator_key = f"{spin}_{num_spin_orbital-Qubit-1} " + operator_key
                if operator_key[:-1] not in data:
                    data[operator_key[:-1]] = coefficient1*coefficient2*0.5
                else :
                    data[operator_key[:-1]] += coefficient1*coefficient2*0.5
        self.n_qubits = num_spin_orbital
        return FermionicOp(data=data, num_spin_orbitals=num_spin_orbital)
        

    def simulate(self, amplitudes):
        """Perform the simulation for the molecule.

        If the mean field is not provided it is automatically calculated.

        Args:
            amplitudes (list): The initial amplitudes (float64).

        Returns:
            float64: The total energy (energy).
        """
        # Test if right number of amplitudes have been passed
        if len(amplitudes) != self.amplitude_dimension:
            raise ValueError("Incorrect dimension for amplitude list.")

        amplitudes = list(amplitudes)
        estimator = AerEstimator(run_options={"shots": None}, approximation=True)

        result = estimator.run(
            circuits=self.uccsd_ansatz, observables=self.jw_hamiltonian, parameter_values=amplitudes
        ).result()

        energy = result.values[0]

        # Update optimal amplitudes
        self.optimized_amplitudes = amplitudes

        return energy


    def get_rdm(self):
        """Obtain the RDMs from the optimized amplitudes.

        Obtain the RDMs from the optimized amplitudes by using the
        same function for energy evaluation.
        The RDMs are computed by using each fermionic Hamiltonian term,
        transforming them and computing the elements one-by-one.
        Note that the Hamiltonian coefficients will not be multiplied
        as in the energy evaluation.
        The first element of the Hamiltonian is the nuclear repulsion
        energy term, not the Hamiltonian term.

        Returns:
            (numpy.array, numpy.array): One & two-particle RDMs (rdm1_np & rdm2_np, float64).
        """

        amplitudes = self.optimized_amplitudes
        one_rdm = np.zeros((self.n_orbitals, self.n_orbitals))
        two_rdm = np.zeros((self.n_orbitals, self.n_orbitals, self.n_orbitals, self.n_orbitals))

        # Loop over all single fermionic hamiltonian term to get RDM values
        all_terms = self.ferm_hamiltonian.terms
        import copy
        fh_copy = copy.deepcopy(self.ferm_hamiltonian)

        for ii in all_terms:
            for jj in ii[1]:
                # Only use a single fermionic term, set its coefficient to 1.
                term_type = ii[0]
                jj = (jj[0], 1.0)
                single_fh = (term_type, [jj])

                fh_copy.terms = [single_fh]
                # Compute qubit Hamiltonian (C# Chemistry library)
                self.jw_hamiltonian = JordanWignerMapper().map(self._QsharpFermionToQiskitEncoding(fh_copy))

                # Compute RDM value
                RDM_value = self.simulate(amplitudes)

                # Update RDM matrices
                ferm_ops = single_fh[1][0][0][0]
                indices = [ferm_op[1] for ferm_op in ferm_ops]

                # 1-RDM matrix
                if (len(term_type) == 2):
                    i, j = indices[0]//2, indices[1]//2
                    if (i == j):
                        one_rdm[i, j] += RDM_value
                    else:
                        one_rdm[i, j] += RDM_value
                        one_rdm[j, i] += RDM_value

                # 2-RDM matrix (works with Microsoft Chemistry library sign convention)
                elif (len(term_type) == 4):
                    i, j, k, l = indices[0]//2, indices[1]//2, indices[2]//2, indices[3]//2

                    if((indices[0]==indices[3]) and (indices[1]==indices[2])):
                        if((indices[0]%2 == indices[2]%2) and (indices[1]%2 == indices[3]%2)):
                            two_rdm[i,l,j,k] += RDM_value
                            two_rdm[j,k,i,l] += RDM_value
                            two_rdm[i,k,j,l] -= RDM_value
                            two_rdm[j,l,i,k] -= RDM_value
                        else:
                            two_rdm[i,l,j,k] += RDM_value
                            two_rdm[j,k,i,l] += RDM_value
                    else:
                        if((indices[0]%2 == indices[3]%2) and (indices[1]%2 == indices[2]%2)):
                            two_rdm[i,l,j,k] += RDM_value
                            two_rdm[j,k,i,l] += RDM_value
                            two_rdm[l,i,k,j] += RDM_value
                            two_rdm[k,j,l,i] += RDM_value
                            if((indices[0]%2 == indices[2]%2) and (indices[1]%2 == indices[3]%2)):
                                two_rdm[i,k,j,l] -= RDM_value
                                two_rdm[j,l,i,k] -= RDM_value
                                two_rdm[k,i,l,j] -= RDM_value
                                two_rdm[l,j,k,i] -= RDM_value
                        else:
                            two_rdm[i,k,j,l] -= RDM_value
                            two_rdm[j,l,i,k] -= RDM_value
                            two_rdm[k,i,l,j] -= RDM_value
                            two_rdm[l,j,k,i] -= RDM_value

        return (one_rdm, two_rdm)

