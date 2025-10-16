import cv2 as cv
import sys
import numpy as np
from numpy import linalg as LA

counter = 0
vertices = []

def afinize(x):
    return x/x[-1]


def normMatrix(points):
    def afinize(x):
        return x/x[-1]
    
    aPts = [afinize(np.array(pt)) for pt in points]
    cog = np.mean(aPts, axis = 0)

    translated = aPts - cog
    G = np.array([[1, 0, -cog[0]], [0, 1, -cog[1]], [0, 0, 1]])

    norms = [LA.norm(pt) for pt in translated]
    avgNorm = np.mean(norms)
    l = np.sqrt(2)/avgNorm
    S = np.array([[l, 0, 0], [0, l, 0], [0, 0, 1]])
    
    return np.array(S@G)

def DLT(org: list, img: list):
    if len(org) != len(img):
        return "Razlicit broj tacaka u slici i originalu!"

    #dve jednacine i pravljenje 2nx9 matrice
    A = []
    for (x1, x2, x3), (xp1, xp2, xp3) in zip(org, img):
        A.append([0, 0, 0, -1*xp3*x1, -1*xp3*x2, -1*xp3*x3, xp2*x1, xp2*x2, xp2*x3])
        A.append([xp3*x1, xp3*x2, xp3*x3, 0, 0, 0, -1*xp1*x1, -1*xp1*x2, -1*xp1*x3])
    A = np.array(A)

    U, D, V = LA.svd(A)
    P = V[-1].reshape(3,3)
    #P = P/P[-1,-1]

    return np.array(P)

def DLTwithNormalization(org, img):
    org = np.array(org)
    img = np.array(img)
    
    T1 = normMatrix(org)
    nOrg = T1 @ org.T
    
    T2 = normMatrix(img)
    nImg = T2 @ img.T

    P2 = DLT(nOrg.T, nImg.T)
    P1 = LA.inv(T2) @ P2 @ T1
    #P1 = P1/P1[-1,-1]

    return np.array(P1)

def draw(event, x, y, flags, param):
    global vertices, counter

    if counter == 4:
        return

    if event == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), 4, (255,80,0), -1)
        cv.putText(img, f'{x}, {y}', (x+5, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,80,0), 1, cv.LINE_AA)
        vertices.append([x,y])
        counter += 1


img = cv.imread('crtez.jpg')
img = cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)

if img is None:
    sys.exit("Nije moguce otvoriti sliku")

cv.namedWindow("Display")
cv.setMouseCallback("Display", draw)

cv.namedWindow("Display2")

transformed = False

while 1:
    cv.imshow('Display', img)

    if transformed:
        cv.imshow('Display2', output)

    if counter == 4:
        if not transformed:
            a,b,c,d = vertices[0], vertices[1], vertices[2], vertices[3]
            ad = np.sqrt((a[0] - d[0])**2 + (a[1] - d[1])**2)
            bc = np.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2)
            width = int((ad + bc) / 2)

            ab = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            cd = np.sqrt((c[0] - d[0])**2 + (c[1] - d[1])**2)
            height = int((ab + cd) / 2)

            final = np.array([[0, 0], [0, height], [width, height], [width,0]])
            vertices = [np.append(x, 1) for x in vertices]
            final = [np.append(x, 1) for x in final]
            transformation = DLTwithNormalization(np.array(vertices), final)

            output = cv.warpPerspective(img, transformation, (width, height), flags=cv.INTER_LINEAR)
            transformed = True

    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()