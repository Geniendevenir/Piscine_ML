import numpy as np
import matplotlib.pyplot as plt


#Create Array
#1 Dimension
A1 = np.array([1, 2, 3, 4])
print(f"A1:\n{A1}\n{A1.shape}\n")

B1 = np.arange(1, 10).astype(float)
print(f"B1:\n{B1}\n{B1.shape}\n")

cShape = (2, 3)
C1 = np.zeros(cShape).astype(float)
print(f"C1:\n{C1}\n{C1.shape}\n")
#Expected:
#	[[0., 0., 0.]
#	[0., 0., 0.]]

#2D
A2 = np.arange(1, 11).reshape(5, 2)
print(f"A2:\n{A2}\n{A2.shape}\n")
#Expected:
#	[[1, 2]
#	[3, 4]
#	[5, 6]
#	[7, 8]
#	[9, 10]]

B2 = np.array("A B C D E F G".split()).reshape(1, 7)
print(f"B2:\n{B2}\n{B2.shape}\n")
#Expected:
# [[A, B, C, D, E, F, G]]

#3D
A3 = np.full((2, 3, 4), 42)
print(f"A3:\n{A3}\n{A3.shape}\n")


#Add a Dimension
#1D to 2D to 3D
D1 = np.arange(10, 15)
print(f"D1:\n{D1}\n{D1.shape}\n")
D2 = np.reshape(D1, (-1, 1)) 						#Reshape Version
print(f"D2 (Reshape):\n{D2}\n{D2.shape}\n")
D2 = D1[:, None] 									#Slicing + None Version
print(f"D2 (Slicing+ None):\n{D2}\n{D2.shape}\n")
D2 = D1[np.newaxis, :]								#Slicing + np.newaxis Version
print(f"D2 (Slicing + np.newaxis):\n{D2}\n{D2.shape}\n")
D3 = np.reshape(D2, (5, 1, 1))						#Reshape Version
print(f"D3 (3D Reshape):\n{D3}\n{D3.shape}\n")

#Add Dimension with another numpy Array
#1D to 2D to 3D
D1 = np.arange(10, 110, 10)
ones = np.ones(D1.shape[0])
D2 = np.stack([ones, D1], axis=0)
print(f"D2 (np.stack(axis=0)):\n{D2}\n{D2.shape}\n")
D2 = np.stack([ones, D1], axis=1)
print(f"D2 (np.stack(axis=1)):\n{D2}\n{D2.shape}\n")

#Create Randomness
#RGB pattern simulations 100x100 shape(100, 100, 3)
rgvShape = (100, 100, 3)
rgb = np.integer(0, 255, size=(rgvShape))
plt.imshow(rgb)
plt.axis("off")
plt.show()

#Shuffle an Array

#Choose a Random Element/Sample

""" 
#Remove a Dimension
E3 = np.ones((1, 1, 5))
E1 = np.squeeze(E3)											#np.Squeeze Version (removes all axis == 1)
print(f"E1 (np.squeeze):\n{E1}\n{E1.shape}\n")				
E2 = np.squeeze(E3, axis=1)									#np.Squeeze + axis Version (removes all len(axis) == 1)
print(f"E2 (np.squeeze(.., axis=)):\n{E2}\n{E2.shape}\n")	
E2 = E3.reshape(1, 5, 1)									#Reshape Version
print(f"E2 (Reshape):\n{E2}\n{E2.shape}\n")
E2 = E3.reshape(1, -1, 1)									#Reshape + Auto-infer  Version (-1 == Auto Detect the logical correct Index)
print(f"E2 (Reshape + Auto-Infer):\n{E2}\n{E2.shape}\n")
E2 = E3[:, :, 0]											#Indexing (Drop the index dimension)
print(f"E2 (Indexing):\n{E2}\n{E2.shape}\n")

#Flatten
F3 = np.zeros((5, 2, 7))
F1 = F3.flatten()
print(f"F1 (.fatten()):\n{F1}\n{F1.shape}\n")		#arr.fatten() Version
F1 = F3.ravel()
print(f"F1 (.ravel()):\n{F1}\n{F1.shape}\n")		#arr.ravel() Version


#Get Values of the Dimension of an Array """