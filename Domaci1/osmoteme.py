import numpy as np

def nevidljivo(ps):

    def afinize(x):
        y = [e/x[-1] for e in x]
        return y
    
    ps = [np.append(p, [1]) for p in ps]

    p1 = ps[4]
    p2 = ps[5]
    p3 = ps[6]
    p5 = ps[0]
    p6 = ps[1]
    p7 = ps[2]
    p8 = ps[3]

    xb1 = np.round(afinize(np.cross(np.cross(p2, p6), np.cross(p1, p5))))
    xb2 = np.round(afinize(np.cross(np.cross(p2, p6), np.cross(p3, p7))))
    xb3 = np.round(afinize(np.cross(np.cross(p1, p5), np.cross(p3, p7))))

    yb1 = np.round(afinize(np.cross(np.cross(p7, p8), np.cross(p6, p5))))
    yb2 = np.round(afinize(np.cross(np.cross(p7, p8), np.cross(p2, p1))))
    yb3 = np.round(afinize(np.cross(np.cross(p6, p5), np.cross(p2, p1))))

    xb = np.round((xb1 + xb2 + xb3) / 3)
    yb = np.round((yb1 + yb2 + yb3) / 3)

    p4 = np.cross(np.cross(p8, xb), np.cross(p3, yb))
    p4 = afinize(p4)
    p4 = np.round(p4, 0)

    return [int(p4[0]), int(p4[1])]

if __name__ == '__main__':

    p1 = [595, 301]
    p2 = [292, 517]
    p3 = [157, 379]
    p5 = [665, 116]
    p6 = [304, 295]
    p7 = [135, 163]
    p8 = [509, 43]

    ps = [p5, p6, p7, p8, p1, p2, p3]
