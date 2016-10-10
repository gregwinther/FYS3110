'''
TAKE HOME MIDTERM EXAM, Quantum Mechanics FYS3110
The first part of this script is to check the computation
in problem 1.4.
'''

import numpy as np 
import scipy.linalg

up = np.array([[1], [0]])
dn = np.array([[0], [1]])

S_plus 	= np.array([[0, 1], [0, 0]])
S_minus = np.array([[0, 0], [1, 0]])
Sz = (1.0/2)*np.array([[1, 0], [0, -1]])

S1z = np.kron(Sz, np.kron(np.eye(2), np.eye(2)))
S2z = np.kron(np.eye(2), np.kron(Sz, np.eye(2)))
S3z = np.kron(np.eye(2), np.kron(np.eye(2), Sz)) 

Sztot = S1z + S2z + S3z

S1_plus = np.kron(S_plus, np.kron(np.eye(2), np.eye(2)))
S2_plus = np.kron(np.eye(2), np.kron(S_plus, np.eye(2)))
S3_plus = np.kron(np.eye(2), np.kron(np.eye(2), S_plus))

S1_minus = np.kron(S_minus, np.kron(np.eye(2), np.eye(2)))
S2_minus = np.kron(np.eye(2), np.kron(S_minus, np.eye(2)))
S3_minus = np.kron(np.eye(2), np.kron(np.eye(2), S_minus))

# Hamilton operator w/o (J/hbar^2) factor
def Hamilton(state):
	return \
	(1.0/2)*\
		(np.dot(S1_plus, np.dot(S2_minus, state)) +\
		np.dot(S2_plus, np.dot(S1_minus,state))) +\
		np.dot(S1z, np.dot(S2z, state)) +\
	(1.0/2)*\
		(np.dot(S2_plus, np.dot(S3_minus, state)) +\
		np.dot(S3_plus, np.dot(S2_minus,state))) +\
		np.dot(S2z, np.dot(S3z, state)) + \
	(1.0/2)*\
		(np.dot(S3_plus, np.dot(S1_minus, state)) +\
		np.dot(S1_plus, np.dot(S3_minus,state))) +\
		np.dot(S3z, np.dot(S1z, state)) 


updndn = np.kron(up, np.kron(dn, dn))
print("Hamiltonian(up down down) = ")
print(Hamilton(updndn))

'''
Computation of probabilities that the state is preserved,
problem 1.8
'''

# By scaling the probem, hbar can be set to one
# The value om J seems somewhat arbitrary
hbar 	= float(1)
J 		= float(1)

Hoperator = (J/(hbar*hbar))*\
				((1.0/2)*(S1_plus*S2_minus + S2_plus*S1_plus) + S1z*S2z +\
				 (1.0/2)*(S2_plus*S3_minus + S3_plus*S2_plus) + S2z*S3z +\
				 (1.0/2)*(S3_plus*S1_minus + S1_plus*S3_plus) + S3z*S1z)

def propagator(t, hbar=hbar, J=J):
	matrixexponential = scipy.linalg.expm((-1.0)*(0+1j)*Hoperator*t/hbar)
	return matrixexponential

# Function that returns a bra from a
def bra(ket):
	return np.transpose(np.conj(ket))

#matrixexponential = scipy.linalg.expm(S1S2)
#print(scipy.linalg.norm(np.dot(matrixexponential, updndn)))
newstate 	= np.dot(propagator(1), updndn)
newstate2 	= np.dot(propagator(2), updndn)
newstate3	= np.dot(propagator(4), updndn)

'''
print("Up down down: ")
print(updndn)
print("Inner prod sq: ", np.dot(bra(updndn), updndn)**2)
print("")
print("New state, t=1: ")
print(newstate)
print("Inner prod sq: ", np.dot(bra(updndn), newstate)**2)
print("")
print("New state, t=100: ")
print(newstate2)
print("Innner prod sq: ", np.dot(bra(updndn), newstate3)**2)
'''
print("\nH = ")
print(Hoperator)
print("\nSztot = ")
print(Sztot)
print("\nH*Sztot-Sztot*H = ")
print(np.dot(Hoperator,Sztot) - np.dot(Sztot,Hoperator))
