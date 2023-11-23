import numpy as np

# Represents a linear programming problem in standard form
class standardLP:
    # Create a lp problem in standard form
    def __init__(self, c, A, b):    
        self.c = np.array(c, dtype=float)[np.newaxis].T
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)[np.newaxis].T

        self.value = 0

    # Solves the lp problem. Sets self.status to "OPTIMAL", "INFEASIBLE", "UNBOUNDED" accordingly
    def solve(self): 
        # Create the basic and nonbasic index sets
        n = self.c.size
        m = self.b.size

        nonbasic = [i for i in range(n)]
        basic = [i for i in range(n + m) if not i in nonbasic]

        # Change A and c to include slack variables
        self.A = np.concatenate((self.A, np.identity(m)), 1)
        self.c = np.concatenate((self.c, np.zeros(m)[np.newaxis].T), 0)

        # Check for infeasibility
        if (np.any(self.b < 0)):
            print("Initial dictionary is infeasible. Performing phase 1.")
            c = self.c 
            self.c = np.zeros((n+m + 1, 1), dtype=float)
            self.c[0] = -1

            self.A = np.concatenate((-1 * np.ones((m, 1), dtype=float), self.A), 1)

            nonbasic = [i for i in range(n + 1)]
            basic = [i for i in range(n + m + 1) if not i in nonbasic]

            eIdx = 0
            maxRatio = np.max(-1 * self.b)
            rowIdx = np.where(-1 * self.b == maxRatio)[0][0]
            lIdx = basic[rowIdx]

            basic = [i for i in basic if i != lIdx] + [eIdx]
            nonbasic = [i for i in nonbasic if i != eIdx] + [lIdx]

            basic.sort()
            nonbasic.sort()

            print(f"x0 entering, x{lIdx} leaving")
            self.pivot(0, lIdx, basic, nonbasic)

            while (np.any(self.c > 0)):
                # Get entering and leaving variables
                eIdx = getEntering(self.c, nonbasic) 

                lIdx = getLeaving(self.A[:,[eIdx]], self.b, basic)
                if (lIdx == -1):
                    self.status = "UNBOUNDED"
                    return

                print(f"x{eIdx} entering, x{lIdx} leaving")

                basic = [i for i in basic if i != lIdx] + [eIdx]
                nonbasic = [i for i in nonbasic if i != eIdx] + [lIdx]

                basic.sort()
                nonbasic.sort()

                # Perform the pivot
                self.pivot(eIdx, lIdx, basic, nonbasic)

            if self.value != 0:
                self.status = "INFEASIBLE"
                return 

            nonbasic = [i - 1 for i in nonbasic if i > 0]
            basic = [i - 1 for i in basic]

            self.c = c 
            self.A = self.A[:,1:]
            self.value = self.c.T[:,basic]@self.b 

            self.c = self.c.T - self.c.T[:,basic]@self.A 
            self.c = self.c.T 
            
        # Simplex algorithm
        while (np.any(self.c > 0)):
            # Get entering and leaving variables
            eIdx = getEntering(self.c, nonbasic) 

            lIdx = getLeaving(self.A[:,[eIdx]], self.b, basic)
            if (lIdx == -1):
                self.status = "UNBOUNDED"
                return

            print(f"x{eIdx + 1} entering, x{lIdx + 1} leaving")

            basic = [i for i in basic if i != lIdx] + [eIdx]
            nonbasic = [i for i in nonbasic if i != eIdx] + [lIdx]

            basic.sort()
            nonbasic.sort()

            # Perform the pivot
            self.pivot(eIdx, lIdx, basic, nonbasic)
            

        # Simplex algorithm terminates and finds optimal 
        self.status = "OPTIMAL"
        self.solution = (self.A, self.b, self.c, self.value,basic,nonbasic)
    
    def pivot(self, eIdx, lIdx, basic, nonbasic):
        B = self.A[:,basic]
        BInvs = np.linalg.inv(B)

        self.A = BInvs@self.A 
        self.b = BInvs@self.b 
        self.value += self.c.T[:,basic]@self.b 

        self.c = self.c.T - self.c.T[:,basic]@self.A 
        self.c = self.c.T 
       

# Returns the index of the entering variable according to Bland's rule
def getEntering(c, nonbasic):
    rows = c[nonbasic]
    return nonbasic[np.where(rows > 0)[0][0]]

# Returns the index of the leaving variable given the column of the entering variable and b
def getLeaving(e, b, basic):
    epsilon = np.max(b[b > 0])/1e+100
    ratio = e/(b + epsilon)
    if (np.all(ratio <= 0)):
        return -1

    maxRatio = np.max(ratio)
    rowIdx = np.where(ratio == maxRatio)[0][0]
    return basic[rowIdx]
