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

S1_plus = np.kron(S_plus, np.kron(np.eye(2), np.eye(2)))
S2_plus = np.kron(np.eye(2), np.kron(S_plus, np.eye(2)))
S3_plus = np.kron(np.eye(2), np.kron(np.eye(2), S_plus))

S1_minus = np.kron(S_minus, np.kron(np.eye(2), np.eye(2)))
S2_minus = np.kron(np.eye(2), np.kron(S_minus, np.eye(2)))
S3_minus = np.kron(np.eye(2), np.kron(np.eye(2), S_minus))

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
print(Hamilton(updndn))

S1S2 =  (1.0/2)*(S1_plus*S2_minus + S2_plus*S1_plus) + S1z*S2z +\
		(1.0/2)*(S2_plus*S3_minus + S3_plus*S2_plus) + S2z*S3z +\
		(1.0/2)*(S3_plus*S1_minus + S1_plus*S3_plus) + S3z*S1z

print(S1S2)
matrixexponential = scipy.linalg.expm(S1S2)
print(scipy.linalg.norm(np.dot(matrixexponential, updndn)))
