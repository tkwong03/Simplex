import numpy as np
import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import simplex

class TestSimplexMethods(unittest.TestCase):
    def test_getEntering(self):
        c = np.array([1,2,0,0])[np.newaxis].T 
        nonbasic = [0,1]
        self.assertEqual(simplex.getEntering(c,nonbasic), 0)

        c = np.array([0,0,1,1,0,0])[np.newaxis].T 
        nonbasic = [2,3]
        self.assertEqual(simplex.getEntering(c,nonbasic), 2)

        c = np.array([0,-1,-2,3,4,0])[np.newaxis].T 
        nonbasic = [1,3,4]
        self.assertEqual(simplex.getEntering(c,nonbasic), 3)

    def test_getLeaving(self):
        e = np.array([1,2,3])[np.newaxis].T 
        b = np.array([1,1,1])[np.newaxis].T
        basic = [0,1,2]
        self.assertEqual(simplex.getLeaving(e,b,basic), 2)

        b = np.array([1,2,3])[np.newaxis].T 
        self.assertEqual(simplex.getLeaving(e,b,basic), 0)

        e = np.array([-1,2,3])[np.newaxis].T 
        self.assertEqual(simplex.getLeaving(e,b,basic), 1)

    def test_solve_optimal(self):
        c = [6,8,5,9]
        b = [5,3]
        A = [[2,1,1,3],
             [1,3,1,2]]

        lp = simplex.standardLP(c, A, b)

        lp.solve()

        A = np.array(
                [[1,-2,0,1,1,-1],
                [0,5,1,1,-1,2]]
                , dtype=float)
        b = np.array([2,1])[np.newaxis].T 
        c = np.array([0,-5,0,-2,-1,-4])[np.newaxis].T 
        basic = [0,2]
        nonbasic = [1,3,4,5]
        value = 17 

        self.assertEqual(lp.status, "OPTIMAL")
        self.assertTrue(np.allclose(lp.solution[0], A))
        self.assertTrue(np.allclose(lp.solution[1], b))
        self.assertTrue(np.allclose(lp.solution[2], c))
        self.assertEqual(lp.solution[3], value)
        self.assertEqual(lp.solution[4], basic)
        self.assertEqual(lp.solution[5], nonbasic)

    def test_solve_infeasible(self):
        c = [-2, -1]
        b = [-1,-2,1]
        A = [[-1,1],
             [-1,-2],
             [0,1]]

        lp = simplex.standardLP(c,A,b)

        lp.solve()

        self.assertEqual(lp.status, "INFEASIBLE")

    def test_solve_unbounded(self):
        c = [1,-1]
        A = [[-2,3],
             [0,4],
             [0,-1]]
        b = [5,7,0]

        lp = simplex.standardLP(c,A,b)
        lp.solve()

        self.assertEqual(lp.status, "UNBOUNDED")

        

if __name__ == '__main__':
    unittest.main()
