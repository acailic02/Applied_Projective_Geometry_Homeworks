import numpy as np
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def center(T):
    C1 = LA.det(np.delete(T,0,1))
    C2 = LA.det(np.delete(T,1,1))
    C3 = LA.det(np.delete(T,2,1))
    C4 = LA.det(np.delete(T,3,1))

    C = np.array([C1, -C2, C3, -C4]) / -C4
    C = np.where(np.isclose(C, 0) , 0.0 , C)

    return C

def camera_K(T):
    T0 = np.delete(T, 3, 1)
    T0i = LA.inv(T0)

    T1 = T0i[:, 0]
    T2 = T0i[:, 1]
    T3 = T0i[:, 2]

    Q1 = T1
    Q2 = T2 - ((np.dot(T2, Q1) / np.dot(Q1, Q1)) * Q1)
    Q3 = T3 - ((np.dot(T3, Q2) / np.dot(Q2, Q2)) * Q2) - ((np.dot(T3, Q1) / np.dot(Q1, Q1)) * Q1)

    Q1 = Q1 / LA.norm(Q1)
    Q2 = Q2 / LA.norm(Q2)
    Q3 = Q3 / LA.norm(Q3)

    K = np.dot(T0, np.column_stack([Q1, Q2, Q3]))
    K = K / K[-1, -1]
    K = np.where(np.isclose(K, 0), 0.0, K)

    return K

def camera_A(T):
    T0 = np.delete(T,3,1)
    T0i = LA.inv(T0)
    
    T1 = T0i[:, 0]
    T2 = T0i[:, 1]
    T3 = T0i[:, 2]
    
    Q1 = T1 
    Q2 = T2 - ((np.dot(T2, Q1) / np.dot(Q1, Q1)) * Q1)
    Q3 = T3 - ((np.dot(T3, Q2) / np.dot(Q2, Q2)) * Q2) - ((np.dot(T3, Q1) / np.dot(Q1, Q1)) * Q1)

    Q1 = Q1 / LA.norm(Q1)
    Q2 = Q2 / LA.norm(Q2)
    Q3 = Q3 / LA.norm(Q3)
    
    A = np.column_stack([Q1, Q2, Q3])
    
    if LA.det(A) < 0:
        A *= -1
    
    A = np.where(np.isclose(A, 0) , 0.0 , A)
    return A

def camera_matrix(pts2D, pts3D):
    n = len(pts3D)
    X = []
 
    for i in range(n):
        A, B, C, D = pts3D[i]
        a, b, d = pts2D[i]
        
        X.append([0, 0, 0, 0, -d*A, -d*B, -d*C, -d*D, b*A, b*B, b*C, b*D])
        X.append([d*A, d*B, d*C, d*D, 0, 0, 0, 0, -a*A, -a*B, -a*C, -a*D])
        
    X = np.array(X)
    
    _, _, V = LA.svd(X)
    T = V[-1].reshape(3, 4)
 
    T = T / T[-1, -1]
    T = np.where(np.isclose(T, 0) , 0.0 , T)
    return T

# img resolution: 2604x4624

org =  np.array([1,-1,-1])*(np.array([9248,0,0]) -  np.array([[4461, 166, 1], [5679, 798, 1], [4560, 1635, 1], [3293, 856, 1],
                                                              [3477, 2062, 1], [4560, 2797, 1], [5502, 2011, 1]]))

img = np.array([[0, 0, 3, 1], [0, 3, 3, 1], [3, 3, 3, 1], [3, 0, 3, 1], [3, 0, 0, 1], [3, 3, 0, 1], [0, 3, 0, 1]]) 
o = np.array([0, 0, 0])

sides = [[img[4][:-1], img[5][:-1], img[6][:-1], o],  
         [img[3][:-1], img[2][:-1], img[1][:-1], img[0][:-1]],  
         [img[4][:-1], img[5][:-1], img[2][:-1], img[3][:-1]],  
         [img[6][:-1], o, img[0][:-1], img[1][:-1]],  
         [img[5][:-1], img[6][:-1], img[1][:-1], img[2][:-1]],  
         [img[4][:-1], o, img[0][:-1], img[3][:-1]]]

b_x = np.array([10, 0, 0])
b_y = np.array([0, 10, 0])
b_z = np.array([0, 0, 10])

T1 = camera_matrix(org, img)
K = camera_K(T1)
A = camera_A(T1)
C = center(T1)

print("Matrica kamere: \n", T1)
print()
print("Matrica kalibracije kamere: \n", K)
print()
print("Pozicija centra kamere: \n", C)
print()
print("Spoljasnja matrica kamere: \n", A)

c_x = A[0]
c_y = A[1]
c_z = A[2]

fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
axs.add_collection3d(Poly3DCollection(sides, facecolors='orchid', linewidths=0.5, edgecolors='black', alpha=1))

axs.quiver(*o, *b_x, color = 'blue')
axs.quiver(*o, *b_y, color = 'green')
axs.quiver(*o, *b_z, color = 'red')

axs.quiver(*C[:3], *c_x, color = 'blue')
axs.quiver(*C[:3], *c_y, color = 'green')
axs.quiver(*C[:3], *c_z, color = 'red')

axs.set_xlabel('X axis')
axs.set_ylabel('Y axis')
axs.set_zlabel('Z axis')
axs.set_xlim([-1, 15])
axs.set_ylim([-1, 15])
axs.set_zlim([0, 15])

plt.show()