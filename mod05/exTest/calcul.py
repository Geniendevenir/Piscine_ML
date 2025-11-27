import numpy as np

Scalar = 10
Vector = np.array([1, 2, 3, 4, 5])
Vector2 = np.array([1, 2, 3])
Matrix = np.array([[2, 4, 6], [8, 10, 12]])
			  
#I - ELEMENT WISE OPERATION
#Addition
#Vector + Scalar
#Vector + Vector
#Matrix + Scalar
#Matrix + Vector
#Matrix + Matrix
print(repr(Matrix + Vector2))

#Sub
#Vector - Scalar
#Vector - Vector
#Matrix - Vector
#Matrix - Matrix

#Div
#Vector / Scalar
#Vector / Vector
#Matrix / Vector
#Matrix / Matrix

#Mult
#Vector * Scalar
#Vector * Vector
#Matrix * Vector
#Matrix * Matrix


#II - Matrix Multiplication
#np.Dot(A, B), @
#Vector @ Scalar
#Vector @ Vector
#Matrix @ Vector
#Matrix @ Matrix


#III - REDUCTION OPERATIONS
#Sum

#Mean

#Max, Min

#IV - BROADCASTING
""" Rules of broadcasting
	-> Two dimensions are compatible if:
	-> They are equal, or
	-> One of them is 1, or
	-> One is missing (treated as 1 on the left) """

#IV - TRANSPOSITION .T



