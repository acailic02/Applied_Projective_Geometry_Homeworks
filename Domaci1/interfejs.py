from osmoteme import nevidljivo
import cv2 as cv
import sys

vertices = []
counter = 0

def draw(event, x, y, flags, param):
    global vertices, counter

    if counter == 7:
        return

    if event == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), 5, (0,255,0), -1)
        cv.putText(img, f'{x}, {y}', (x+5, y), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 5, cv.LINE_AA)
        vertices.append([x,y])
        counter += 1
        

img = cv.imread('rubikova16_9.jpg')
 
if img is None:
    sys.exit("Nije moguce otvoriti sliku")

cv.namedWindow("Display", cv.WINDOW_NORMAL)
cv.setMouseCallback("Display", draw)

while 1:
    cv.imshow('Display', img)

    if counter == 7:
        p4 = nevidljivo([vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6]])
        print(p4)
        x, y = int(p4[0]), int(p4[1])
        cv.circle(img, (x, y), 5, (0,255,0), -1)
        cv.putText(img, f'{x}, {y}', (x+5, y), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 5, cv.LINE_AA)

    if cv.waitKey(0) | 0xFF == 27:
        break

cv.destroyAllWindows()
