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

"""Perform quantum simulation based on VQE algorithm.

The electronic structure calculation employing the
quantum/classical hybrid variational quantum eigensolver
(VQE) algorithm is done here.
The quantum eigensolver runs inside the classical optimizer.

There are options for which hardware backend can be used.

"""

import warnings
import itertools

import numpy as np
from pyscf import scf

from openqemist.quantum_solvers.initial_parameters import mp2_initial_amplitudes

from ..electronic_structure_solver import ElectronicStructureSolver
from functools import partial

class VQESolver(ElectronicStructureSolver):
    """Estimates energy wih a variational quantum eigensolver algorithm.

    Uses the VQE algorithm to solve the electronic structure problem. By default
    an optimizer from scipy is used, but users can set any function whose first
    argument is the `simulate` function of the hardware backend and the second
    is the amplitudes to optimize over and returns a energy. See the
    implementation of `_default_optimizer` for a concrete example.
    Users should provide a hardware backend type that conforms to the interface
    of `openqemist.quantum_solver.ParametricQuantumSolver` that the `VQESolver`
    will construct and use. Users should also provide an ansatze type that is
    supported by the backend.
    Users can also provide a function that takes a `pyscf.gto.Mole` as its first
    argument and `pyscf.scf.RHF` as is second and returns the inital amplitudes
    for the variational optimization. The user is responsible for ensuring that
    the dimension of the amplitudes vector is correct for the given molecule and
    andsatz choice.

    Attributes:
        hardware_backend_type (subclass of ParametricQuantumSolver): A type for
            the backend instance that is automatically constructed.
        ansatz_type (subclass of Enum): Type of ansatz that is supported by the
            backend.
        optimizer (function): Function that is called to optimize.
        initial_amplitude_function (function): Function that returns the initial
            amplitudes used for the optimization.
        initial_amplitudes (list): The initial amplitudes for the optimzation.
        verbose (boolean): Controls the verbosity of the default optimizer.
    Note:
        Initial amplitudes can be specified both through the functional
        parameter `initial_amplitude_function` and through the
        `initial_amplitudes`. If both are specified the `initial_amplitudes` are
        used.
    """

    def __init__(self):
        self.verbose = True
        self.hardware_backend_type = None
        self.hardware_backend = None
        self.ansatz_type = None
        self.optimizer = None
        self.initial_amplitude_function = None
        self.initial_amplitudes = None

    def simulate(self, molecule, mean_field=None):
        """Perform the simulation for the molecule.

        If the mean field is not provided it is automatically calculated.

        Args:
            molecule (pyscf.gto.Mole): The molecule to simulate.
            mean_field (pyscf.scf.RHF): The mean field of the molecule.
        """
        # Calculate the mean field if the user has not already done it.
        if not mean_field:
            mean_field = scf.RHF(molecule)
            mean_field.verbose = 0
            mean_field.scf()

        # Check the convergence of the mean field
        if not mean_field.converged:
            warnings.warn("VQESolver simulating with mean field not converged.",
                    RuntimeWarning)

        # Set up the instance of the hardware backend
        if self.verbose:
            print("VQE : Setting up hardware backend\n")
        self.hardware_backend = self.hardware_backend_type(self.ansatz_type,
                molecule, mean_field,self.verbose)

        # If no set of initial amplitudes was provided, set them as MP2 amplitudes
        if self.initial_amplitudes:
            amplitudes = self.initial_amplitudes
        elif self.initial_amplitude_function:
            amplitudes = self.initial_amplitude_function()
        else:
            amplitudes = self.random_initial_amplitudes()
        if self.verbose:
            print("VQE : initial amplitudes\n", amplitudes, "\n\n")

        # If the user didn't provide an optimizer, then we give them scipy's
        if not self.optimizer:
            self.optimizer = self._default_optimizer

        # Call the optimizer until it converges
        energy = self.optimizer(self.hardware_backend.simulate, amplitudes)

        return energy

    def get_rdm(self):
        """Returns the RDM from the hardware backend.

        Returns the reduced density matrices from the hardware backend. Does not
        catch and exceptions that the hardware backend raises if it is not in a
        state to return the RDM.

        Returns:
            (numpy.array,numpy.array): One & two-particle RDMs (float64).

        Raises:
            RuntimeError: If no simulation has been run.
        """
        if not self.hardware_backend:
            raise RuntimeError("Cannot retrieve RDM because no simulation has been run.")

        return self.hardware_backend.get_rdm()

    def _default_optimizer(self, backend, amplitudes):
        """Function that can be set as a default optimizer.

        Funciton that is used by the class as a default optimizer when user does
        not provide one.

        Args:
            backend (ParametricSolver): The quantum solver.
            amplitudes (list): The variational parameters (float64).

        Returns:
            list: The new variational parameters (result.fun, float64).
        """

        from scipy.optimize import minimize

        result = minimize(backend, amplitudes, method='COBYLA',
                options={'disp':True, 'maxiter':2000, 'rhobeg':0.01, 'tol':1e-3})

        if self.verbose:
            print("\n\t\tOptimal UCCSD Singlet Energy: {}".format(result.fun))
            print("\t\tOptimal UCCSD Singlet Amplitudes: {}".format(result.x))
            print("\t\tNumber of Function Evaluations : ", result.nfev)

        return result.fun
    
    def random_initial_amplitudes(self):
        return np.random.rand(self.hardware_backend.amplitude_dimension)
