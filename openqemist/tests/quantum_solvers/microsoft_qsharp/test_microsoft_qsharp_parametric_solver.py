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

import unittest
import numpy as np

from pyscf import gto, scf, mp

from openqemist.quantum_solvers import MicrosoftQSharpParametricSolver

def get_file_path_stub():
    """ Gets the path of the test files from anywhere in the test tree."

    The direcory structure should be $SOMETHING/QEMIST/qemist/tests/$SOMETHINGELSE
    so we trim after "tests", then add the path to the results files so we can
    run the tests from anywhere in the tree."""
    import os
    cwd = os.getcwd()
    tests_root = cwd[0:cwd.find("tests") + 5]
    return tests_root + "/quantum_solvers/microsoft_qsharp/data/"

def matricize_2rdm(two_rdm, n_electrons, n_orbitals):
    """ Turns the two_rdm tensor into a matrix for test purposes """
    
    l = 0
    sq = n_orbitals*n_orbitals
    jpqrs = np.zeros((n_orbitals,n_orbitals),dtype=np.int64)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            jpqrs[i,j] = l
            l += 1

    rho = np.zeros((sq,sq))
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            ij = jpqrs[i,j]
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    kl = jpqrs[k,l]
                    rho[ij,kl] += two_rdm[i,k,j,l]

    return rho


H2 = """
        H       0.00        0.00        0.0
        H       0.00        0.00        0.74137727
"""

H4 = """
    H   0.7071067811865476   0.0                 0.0
    H   0.0                  0.7071067811865476  0.0
    H  -1.0071067811865476   0.0                 0.0
    H   0.0                 -1.0071067811865476  0.0
"""

LiH = """
    H       0.00        0.00        0.0
    Li      0.00        0.00        1.0
"""

# References for H2
amplitudes_H2 = [1.69971474e-05, 5.65855806e-02]

# References for H4
amplitudes_H4 = [-3.00520142e-05, -3.41547577e-05,  7.61837556e-06 ,-2.24075399e-05,
                 1.12512690e-02,  3.42703399e-01,  3.44523818e-02,  1.46586868e-02,
                 7.69034155e-02,  7.99964875e-03, -1.81430817e-01, -1.06611015e-01,
                 1.12938142e-02, -3.75164050e-02]

# References for LiH
amplitudes_LiH = [-4.17776465e-04, -3.01636877e-02,  2.59247846e-06, -1.81380838e-06,
                   2.56473288e-06, -1.67351123e-06,  4.99995120e-04,  4.87411549e-04,
                   1.67454873e-03,  1.12528808e-02,  8.90183149e-04,  1.82504586e-02,
                   8.40833525e-04,  1.82672779e-02,  3.93722603e-04,  4.83775296e-02,
                   4.99457737e-04, -1.34326076e-18, -3.13927118e-08, -7.69532977e-09,
                  -7.69532977e-09,  1.61799005e-03, -4.24862234e-03, -7.69532977e-09,
                  -3.13927118e-08, -1.07614392e-07, -3.13927118e-08, -1.49223410e-03,
                  -4.35798373e-02,  2.98149476e-03, -3.13927118e-08, -2.66454784e-08,
                  -7.69532977e-09, -7.69532977e-09, -5.64104758e-08, -7.69532977e-09,
                  -1.15815072e-08, -7.69532977e-09,  2.98287967e-03, -3.13927118e-08,
                  -3.13927118e-08, -3.13927118e-08, -1.07614392e-07,  2.94725869e-03]

class MicrosoftQSharpParametricSolverTest(unittest.TestCase):

    def test_no_mf_H2(self):
        """ Tests number of amplitudes as well as simulate and get_RDM methods """

        mol = gto.Mole()
        mol.atom = H2
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        ansatz = MicrosoftQSharpParametricSolver.Ansatze.UCCSD
        solver = MicrosoftQSharpParametricSolver(ansatz, mol)

        # Test that the number of variational parameters is as expected
        self.assertEqual(solver.amplitude_dimension, 2)

        # Test "simulate"
        energy = solver.simulate(amplitudes_H2)
        self.assertAlmostEqual(energy, -1.13727, delta=1e-5)

        # Compute RDM matrices
        one_rdm, two_rdm = solver.get_rdm()

        # Test traces of matrices
        n_elec, n_orb = mol.nelectron, mol.nao_nr()
        rho = matricize_2rdm(two_rdm, n_elec, n_orb)
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg='Trace of one_rdm does not match number of electrons', delta=1e-6)
        self.assertAlmostEqual(np.trace(rho), n_elec*(n_elec-1), msg='Trace of two_rdm does not match n_elec * (n_elec-1)', delta=1e-6)
            
    def test_no_mf_H4(self):
        """ Tests number of amplitudes as well as simulate and get_RDM methods """

        mol = gto.Mole()
        mol.atom = H4
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        # Initialize mean field object with PySCF
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
        twoint = mf._eri
        oneint = mf.get_hcore()
        fock = mf.get_fock()
        
        ansatz = MicrosoftQSharpParametricSolver.Ansatze.UCCSD
        solver = MicrosoftQSharpParametricSolver(ansatz, mol)

        # Test that the number of variational parameters is as expected
        self.assertEqual(solver.amplitude_dimension, 14)

        # Test "simulate"
        energy = solver.simulate(amplitudes_H4)

        # Compute RDM matrices
        one_rdm, two_rdm = solver.get_rdm()

        # Test traces of matrices
        n_elec, n_orb = mol.nelectron, mol.nao_nr()
        rho = matricize_2rdm(two_rdm, n_elec, n_orb)
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg='Trace of one_rdm does not match number of electrons', delta=1e-6)
        self.assertAlmostEqual(np.trace(rho), n_elec*(n_elec-1), msg='Trace of two_rdm does not match n_elec * (n_elec-1)', delta=1e-6)

    def test_no_mf_LiH(self):
        """ Tests get_RDM methods: assume energy is correct and reconstruct from RDM """
        mol = gto.Mole()
        mol.atom = LiH
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()

        # Initialize mean field object with PySCF
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.scf()
        twoint = mf._eri
        oneint = mf.get_hcore()
        fock = mf.get_fock()
                
        ansatz = MicrosoftQSharpParametricSolver.Ansatze.UCCSD
        solver = MicrosoftQSharpParametricSolver(ansatz, mol)

        # Test that the number of variational parameters is as expected
        self.assertEqual(solver.amplitude_dimension, 44)

        # Test "simulate"
        energy = solver.simulate(amplitudes_LiH)

        # Compute RDM matrices
        one_rdm, two_rdm = solver.get_rdm()

        # Test traces of matrices                                                                                                                                               
        n_elec, n_orb = mol.nelectron, mol.nao_nr()
        rho = matricize_2rdm(two_rdm, n_elec, n_orb)
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg='Trace of one_rdm does not match number of electrons', delta=1e-6)
        self.assertAlmostEqual(np.trace(rho), n_elec*(n_elec-1), msg='Trace of two_rdm does not match n_elec * (n_elec-1)', delta=1e-6)

    def test_mf_H2(self):
        """Tests that all the values are set correctly in the constructor."""
        mol = gto.Mole()
        mol.atom = H2
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()
        mean_field = scf.RHF(mol)
        mean_field.verbose = 0
        mean_field.scf()

        ansatz = MicrosoftQSharpParametricSolver.Ansatze.UCCSD
        solver = MicrosoftQSharpParametricSolver(ansatz, mol, mean_field)

        self.assertEqual(solver.amplitude_dimension, 2)

        energy = solver.simulate(amplitudes_H2)
        self.assertAlmostEqual(energy, -1.13727, delta=1e-5)

    def test_simulate_dimension_throw(self):
        """Tests that all the values are set correctly in the constructor."""
        mol = gto.Mole()
        mol.atom = H2
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = 0
        mol.build()
        mean_field = scf.RHF(mol)
        mean_field.verbose = 0
        mean_field.scf()

        ansatz = MicrosoftQSharpParametricSolver.Ansatze.UCCSD
        solver = MicrosoftQSharpParametricSolver(ansatz, mol, mean_field)

        # solver.amplitude_dimension = 2, this should throw.
        self.assertRaises(ValueError, solver.simulate, [0])

if __name__ == "__main__":
    unittest.main()
