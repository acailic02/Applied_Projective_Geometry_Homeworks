import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p1l = [4461, 166]
p2l = [5679, 798]
p3l = [4560, 1635]
p4l = [3293, 856]
p5l = [3477, 2062]
p6l = [4560, 2797]
p7l = [5502, 2011]
p8l = [4472, 1353]

q1l = [7329, 358]
q2l = [8880, 1115]
q3l = [7711, 2209]
q4l = [6047, 1220]
q5l = [5913, 1884]
q6l = [7379, 2861]
q7l = [8540, 1730]
q8l = [7078, 1033]

p1d = [2910, 281]
p2d = [3902, 881]
p3d = [2599, 1635]
p4d = [1622, 932]
p5d = [2068, 2062] 
p6d = [2939, 2778]
p7d = [4029, 2018]
p8d = [3157, 1424]

q1d = [5877, 453]
q2d = [7180, 1169]
q3d = [5736, 2216]
q4d = [4412, 1277]
q5d = [4426, 1909]
q6d = [5630, 2829]
q7d = [7010, 1775]
q8d = [5786, 1070]

lall = [p1l, p2l, p3l, p4l, p5l, p6l, p7l, p8l, q1l, q2l, q3l, q4l, q5l, q6l, q7l, q8l]
rall = [p1d, p2d, p3d, p4d, p5d, p6d, p7d, p8d, q1d, q2d, q3d, q4d, q5d, q6d, q7d, q8d]

#img resolution: 9248 x 5204

lall = [np.array([9248 - x[0], x[1], 1]) for x in lall]
rall = [np.array([9248 - x[0], x[1], 1]) for x in rall]

def equation(left, right):
    return [a * b for a in left for b in right]

matrixForm = [equation(l, r) for l, r in zip(lall, rall)]

U, D, V = LA.svd(matrixForm)
F = np.array(V[-1])
F = F.reshape(3, 3).T
print("Fundamentalna matrica: \n", F)
print()
print("Det F = ", LA.det(F))

U, D, V = LA.svd(F)
e1 = np.array(V[-1]) 
e1 = e1 * (1 / e1[-1])

e2 = np.array(U.T[-1]) 
e2 = e2 * (1 / e2[-1]) 

print("e1: ", e1)
print("e2: ", e2)
print()

D = np.diag(D)
D1 = np.diag([1, 1, 0])
D1 = D1 @ D
F1 = (U @ D1) @ V
print("Det F1 = ", LA.det(F1))


K1 = np.array(
    [[7600, 0, 4624],
    [0, 7500, 2602],
    [0, 0, 1,]])

E = (K1.T @ F1) @ K1

print("Osnovna matrica: \n", E)
print()

Q0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
E0 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

U, SS, V = np.linalg.svd(-E)

EC = (U @ E0) @ U.T
AA = (U @ Q0.T) @ V

print("Matrica EC: \n", EC)
print()
print("Matrica A: \n", AA)  
print()

#pozicija prve kamere u koordinantnom sistemu druge kamere
C = [EC[2, 1], EC[0, 2], EC[1, 0]]

# T2 = K1|0
T2 = np.hstack((K1, np.zeros((K1.shape[0], 1))))
C1 = -AA.T @ C

T1 = (np.row_stack(((K1 @ AA.T).T, (K1 @ C1)))).T
print("Matrica prve kamere T1: \n", T1)
print()
print("Matrica druge kamere T2: \n", T2)
print()



def equations(T1, T2, m1, m2):
    return np.array([
        m1[1]*T1[2] - m1[2]*T1[1],
        -m1[0]*T1[2] + m1[2]*T1[0],
        m2[1]*T2[2] - m2[2]*T2[1],
        -m2[0]*T2[2] + m2[2]*T2[0]
    ])

affine = [] 
print("3D koordinate:\n")
for m1, m2 in zip(lall, rall):
    system = equations(T1, T2, m1, m2)
    _, _, V = LA.svd(system)
    tmp = np.array(V[-1])
    #afine 
    tmp = tmp[:-1] / tmp[-1]
    tmp = tmp * 10
    affine.append(tmp)
    print(tmp)


sides1 = [[affine[0], affine[1], affine[2], affine[3]],
                [affine[2], affine[5], affine[4], affine[3]],
                [affine[1], affine[6], affine[5], affine[2]],
                [affine[0], affine[7], affine[6], affine[1]],
                [affine[0], affine[3], affine[4], affine[7]],
                [affine[7], affine[4], affine[5], affine[6]]
                ]


sides2 = [[affine[8], affine[9], affine[10], affine[11]], 
                [affine[10], affine[13], affine[12], affine[11]], 
                [affine[9], affine[14], affine[13], affine[10]], 
                [affine[8], affine[15], affine[14], affine[9]], 
                [affine[8], affine[11], affine[12], affine[15]], 
                [affine[15], affine[12], affine[13], affine[14]]
                ]

x_c = AA.T[0]
y_c = AA.T[1]
z_c = AA.T[2]

figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.add_collection3d(Poly3DCollection(sides1, facecolors='blue', linewidths=0.5, edgecolors='black', alpha=0.4))
axes.add_collection3d(Poly3DCollection(sides2, facecolors='red', linewidths=0.5, edgecolors='black', alpha=0.4))

axes.quiver(*C, *x_c, color = 'blue')
axes.quiver(*C, *y_c, color = 'green')
axes.quiver(*C, *z_c, color = 'red')


axes.set_xlabel('X osa')
axes.set_ylabel('Y osa')
axes.set_zlabel('Z osa')

axes.set_xlim([-6, 2])
axes.set_ylim([-6, 2])
axes.set_zlim([-6, 1])

plt.show()