import numpy as np
from numpy import linalg as LA
import math
np.set_printoptions(precision=5, suppress=True) 
 
 # ovde pišete pomocne funkcije
 
def matricaKamere(pts2D, pts3D):
 
 # vaš kod
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
 
    T /= T[-1, -1]
    T = np.where(np.isclose(T, 0) , 0.0 , T)
    return T

if __name__ == 'main':
    pts2D = np.array([[12, 61, 31], [1, 95, 4], [20, 82, 19], [56, 50, 55], [32, 65, 84], [46, 39, 16], [67, 63, 78]])
    pts3D = np.array([[44, 61, 31, 99], [17, 84, 40, 45], [20, 59, 65, 3], [37, 81, 70, 82], [7, 95, 8, 29], [31, 61, 91, 37], [82, 99, 80, 7]])
    print(matricaKamere(pts2D,pts3D))
