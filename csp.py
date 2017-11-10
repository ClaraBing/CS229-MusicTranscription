import numpy as np

## Taken from CS221 scheduling starter code

# General code for representing a weighted CSP (Constraint Satisfaction Problem).
# All variables are being referenced by their index instead of their original
# names.
class CSP:
    def __init__(self):
        # Total number of variables in the CSP.
        self.numVars = 0

        # The list of variable names in the same order as they are added. A
        # variable name can be any hashable objects, for example: int, str,
        # or any tuple with hashtable objects.
        self.variables = []

        # Each key K in this dictionary is a variable name.
        # values[K] is the list of domain values that variable K can take on.
        self.values = {}

        # Each entry is a unary factor table for the corresponding variable.
        # The factor table corresponds to the weight distribution of a variable
        # for all added unary factor functions. If there's no unary function for
        # a variable K, there will be no entry for K in unaryFactors.
        # E.g. if B \in ['a', 'b'] is a variable, and we added two
        # unary factor functions f1, f2 for B,
        # then unaryFactors[B]['a'] == f1('a') * f2('a')
        self.unaryFactors = {}

        # Each entry is a dictionary keyed by the name of the other variable
        # involved. The value is a binary factor table, where each table
        # stores the factor value for all possible combinations of
        # the domains of the two variables for all added binary factor
        # functions. The table is represented as a dictionary of dictionary.
        #
        # As an example, if we only have two variables
        # A \in ['b', 'c'],  B \in ['a', 'b']
        # and we've added two binary functions f1(A,B) and f2(A,B) to the CSP,
        # then binaryFactors[A][B]['b']['a'] == f1('b','a') * f2('b','a').
        # binaryFactors[A][A] should return a key error since a variable
        # shouldn't have a binary factor table with itself.

        self.binaryFactors = {}

    def add_variable(self, var, domain):
        """
        Add a new variable to the CSP.
        """
        if var in self.variables:
            raise Exception("Variable name already exists: %s" % str(var))

        self.numVars += 1
        self.variables.append(var)
        self.values[var] = domain
        self.unaryFactors[var] = None
        self.binaryFactors[var] = dict()


    def get_neighbor_vars(self, var):
        """
        Returns a list of variables which are neighbors of |var|.
        """
        return self.binaryFactors[var].keys()

    def add_unary_factor(self, var, factorFunc):
        """
        Add a unary factor function for a variable. Its factor
        value across the domain will be *merged* with any previously added
        unary factor functions through elementwise multiplication.

        How to get unary factor value given a variable |var| and
        value |val|?
        => csp.unaryFactors[var][val]
        """
        factor = {val:float(factorFunc(val)) for val in self.values[var]}
        if self.unaryFactors[var] is not None:
            assert len(self.unaryFactors[var]) == len(factor)
            self.unaryFactors[var] = {val:self.unaryFactors[var][val] * \
                factor[val] for val in factor}
        else:
            self.unaryFactors[var] = factor

    def add_binary_factor(self, var1, var2, factor_func):
        """
        Takes two variable names and a binary factor function
        |factorFunc|, add to binaryFactors. If the two variables already
        had binaryFactors added earlier, they will be *merged* through element
        wise multiplication.

        How to get binary factor value given a variable |var1| with value |val1|
        and variable |var2| with value |val2|?
        => csp.binaryFactors[var1][var2][val1][val2]
        """
        # never shall a binary factor be added over a single variable
        try:
            assert var1 != var2
        except:
            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            print '!! Tip:                                                                       !!'
            print '!! You are adding a binary factor over a same variable...                  !!'
            print '!! Please check your code and avoid doing this.                               !!'
            print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            raise

        self.update_binary_factor_table(var1, var2,
            {val1: {val2: float(factor_func(val1, val2)) \
                for val2 in self.values[var2]} for val1 in self.values[var1]})
        self.update_binary_factor_table(var2, var1, \
            {val2: {val1: float(factor_func(val1, val2)) \
                for val1 in self.values[var1]} for val2 in self.values[var2]})

    def update_binary_factor_table(self, var1, var2, table):
        """
        Private method you can skip for 0c, might be useful for 1c though.
        Update the binary factor table for binaryFactors[var1][var2].
        If it exists, element-wise multiplications will be performed to merge
        them together.
        """
        if var2 not in self.binaryFactors[var1]:
            self.binaryFactors[var1][var2] = table
        else:
            currentTable = self.binaryFactors[var1][var2]
            for i in table:
                for j in table[i]:
                    assert i in currentTable and j in currentTable[i]
                    currentTable[i][j] *= table[i][j]


class PitchContour(CSP):
    # Laplace distribution evaluation for a change of pitch period between two frames.
    # Enforces temporal continuity between two adjacent states.
    def laplaceDistribution(delta, mu=0.4, sigma=2.4):
        return 1.0 / (2 * sigma) * np.exp(-abs(delta - mu)/ sigma)

    # Normal distribution evaluation for a given value, mean and standard deviation
    def gaussianDistribution(x, mu, sigma):
        return 1.0 / math.sqrt(2 * math.pi * sigma * sigma) * \
            math.exp(-1.0/ (2 * sigma**2) * (x-mu)**2)

    # Gaussian Mixture Model evaluation for a state given the K most probables frequencies
    # observed by the CNN.
    # Input: Integer K,
    # probabilities: array of size K of the normalized probabilities for the K most probable frequencies
    # given by the CNN.
    # frequencies: array of size K with the values of the frequencies
    # variances: array of size K with the values of the standard deviation
    def gaussianMixtureModelDistribution(observation, K, probabilities, frequencies, variances):
        normals = [gaussianDistribution(observation, frequencies[i], variances[i]) for i in range(K)]
        return np.sum(np.multiply(normals, probabilities))

    def __init__(self):
        super(PitchContour, self).__init__()

    # Set range of frequencies that can be assigned to the notes
    def setNoteRanges(ranges):
        self.ranges = ranges

    # Set number of notes to be determined in the pitch contour.
    # Probabilities is a N x K matrix, each line represents the probabilities for
    # the K most probable pitches for the note.
    # Frequencies is N x K
    # Variances is N x K
    def setNotes(N, K, probabilities, frequencies, variances):
        self.K = K
        self.probabilities = probabilities
        self.frequencies = frequencies
        self.variances = variances
        for i in range(N):
            self.add_variable(i, self.ranges)
            self.add_unary_factor(i,\
                lambda v : gaussianDistribution(v, K, probabilities[i], frequencies[i], variances[i]))
            if i > 0:
                self.add_binary_factor(i, i-1, \
                    lambda vafter, vbefore: laplaceDistribution(abs(vafter-vbefore)))
