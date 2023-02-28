import os
a = os.path.abspath(".").rfind('/')
projectPath = os.path.abspath(".")[:a+1]

hadamardMatricesPath = 'src/helpers/Hadamard Matrices/'