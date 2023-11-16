import numpy as np

# Represents a linear programming problem in standard form
class standardLP:
    # Create a lp problem in standard form
    def __init__(self, c, A, b):    
        self.c = np.array(c, dtype=float)[np.newaxis].T
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)[np.newaxis].T

    # Solves the lp problem. Assumes the problem has an optimal
    def solve(self): 
        # Create the basic and nonbasic index sets
        n = self.c.size
        m = self.b.size

        nonbasic = [i for i in range(n)]
        basic = [i for i in range(n + m) if not i in nonbasic]

        # Change A and c to include slack variables
        A = np.concatenate((self.A, np.identity(m)), 1)
        c = np.concatenate((self.c, np.zeros(m)[np.newaxis].T), 0)
        b = self.b

        B = A[:,basic]
        BInvs = np.linalg.inv(B)
        N = A[:,nonbasic]

        value = 0

        # Simplex algorithm
        while (np.any(c > 0)):
            # Get entering and leaving variables
            eIdx = getEntering(c, nonbasic) 
            lIdx = getLeaving(A[:,[eIdx]], b, basic)
            print(f"x{eIdx + 1} entering, x{lIdx + 1} leaving")

            basic = [i for i in basic if i != lIdx] + [eIdx]
            nonbasic = [i for i in nonbasic if i != eIdx] + [lIdx]

            basic.sort()
            nonbasic.sort()

            # Update the dictionary
            
            B = A[:,basic]
            BInvs = np.linalg.inv(B)

            A = BInvs@A 
            b = BInvs@b 
            value += c.T[:,basic]@b 

            c = c.T - c.T[:,basic]@A 
            c = c.T 


        return A, b, c, value, basic, nonbasic



# Returns the index of the entering variable according to Bland's rule
def getEntering(c, nonbasic):
    rows = c[nonbasic]
    return nonbasic[np.where(rows > 0)[0][0]]

# Returns the index of the leaving variable given the column of the entering variable and b
def getLeaving(e, b, basic):
    epsilon = np.max(b[b > 0])/1e+100
    ratio = e/(b + epsilon)

    maxRatio = np.max(ratio)
    rowIdx = np.where(ratio == maxRatio)[0][0]
    return basic[rowIdx]
