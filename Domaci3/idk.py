import numpy as np
from numpy import linalg as LA

T = np.array([[-2, 3, 0, 7],
              [-3, 0, 3, -6],
              [1, 0, 0, -2]])

def center(T):
    C1 = LA.det(np.delete(T,0,1))
    C2 = LA.det(np.delete(T,1,1))
    C3 = LA.det(np.delete(T,2,1))
    C4 = LA.det(np.delete(T,3,1))

    C = np.array([C1, -C2, C3, -C4]) / -C4

    return C

print(center(T))

def cameraK(T):
    T0 = np.delete(T, 3, 1)
    T0i = LA.inv(T0)

    T1 = T0i[:, 0]
    T2 = T0i[:, 1]
    T3 = T0i[:, 2]

    Q1 = T1
    Q1n = LA.norm(Q1)

    if Q1n != 0:
        Q1 /= Q1n

    Q2 = T2 - ((np.dot(T2, Q1) / np.dot(Q1, Q1)) * Q1)
    Q2n = LA.norm(Q2)

    if Q2n != 0:
        Q2 /= Q2n

    Q3 = T3 - ((np.dot(T3, Q2) / np.dot(Q2, Q2)) * Q2) - ((np.dot(T3, Q1) / np.dot(Q1, Q1)) * Q1)
    Q3n = LA.norm(Q3)

    if Q3n != 0:
        Q3 /= Q3n

    K = np.dot(T0, np.column_stack([Q1, Q2, Q3]))
    K = np.where(np.isclose(K, 0), 0.0, K)

    return K / K[-1, -1]

print(cameraK(T))


 
def kameraA(T):
 
 # vaš kod
    T0 = np.delete(T,3,1)
    T0i = LA.inv(T0)
    
    T1 = T0i[:, 0]
    T2 = T0i[:, 1]
    T3 = T0i[:, 2]
    
    Q1 = T1 / LA.norm(T1)
    
    Q2 = T2 - ((np.dot(T2, Q1) / np.dot(Q1, Q1)) * Q1)
    Q2 /= LA.norm(Q2)
    
    Q3 = T3 - ((np.dot(T3, Q2) / np.dot(Q2, Q2)) * Q2) - ((np.dot(T3, Q1) / np.dot(Q1, Q1)) * Q1)
    Q3 /= LA.norm(Q3)
    
    A = np.column_stack([Q1, Q2, Q3])
    
    if LA.det(A) < 0:
        A *= -1
    
    A = np.where(np.isclose(A, 0) , 0.0 , A)
    return A
 
print(kameraA(T))