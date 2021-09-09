# Antes de la competencia, ajustar los rangos de colores HSV con ayuda de los controles deslizables (sliders)
# para obtener una máscara nítida en función de los colores del entorno que capture la camara del Zagi.

import cv2
import numpy as np
from stack_images import stackImages

# Rangos HSV
lowerGreen = np.array([40, 72, 59])
upperGreen = np.array([77, 255, 255])
lowerPink = np.array([255, 255, 255])
upperPink = np.array([255, 255, 255])

listContour = []
listHorizontal = []
listVertical = []
cap = cv2.VideoCapture(0)
#imgobs = cv2.imread('images/3.jpeg')

# Asigna resolusion de la camara
cap.set(3, 480)
cap.set(4, 320)
_, frame = cap.read()
row, cols, _ = frame.shape
centerHorizontal = cols // 2
centerVertical = row // 2

# ----- Función que clasifica y ordena los contornos por color, tambien hace el cálculo por donde tiene que pasar el Zagi -----
def getContours(imgContour, img, label):

    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>1000:
            cv2.drawContours(imgContour, cnt, -1, (0, 0, 0), 5)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),5)
            if w > h:
                position = True
            else:
                position = False

            #Coloca el nombre del color del contorno
            cv2.putText(imgContour,str(position),
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,1,
                        (0,255,255),2)

            listData = (position, x, y, w, h)

            #listado de tuplas (posiciones de contornos)
            listContour.append(listData)

            #listado de contornos horizontales
            if listData[0]:
                listHorizontal.append(listData)
            else:
                listVertical.append(listData)

    #Ordena todos los contornos en funcion de las coordenadas (x, y)
    listContour.sort(key=lambda x: x[4])
    listHorizontal.sort(key=lambda x: x[1] + x[2])
    listVertical.sort(key=lambda x: x[1] + x[2])
    #print(listContour)
    try:
        #-----------------------------------------------------------------------------------------------------------------------
        #ch = sum(listHorizontal)/float(len(listHorizontal))
        # Variables de mensajes
        message = ''
        message_obs = ''

        # Variables de correccion
        correction_horizontal = 0
        correction_vertical = 0

        # Coordenadas obstaculo
        x = listHorizontal[0][1]
        y = listHorizontal[-1][2]
        w = listVertical[-1][1] - listHorizontal[0][1]
        h = listVertical[-1][4] - (listHorizontal[0][2] - listVertical[0][2])
        half_vertical_total = listVertical[-1][2] + listVertical[-1][4] // 2
        half_horizontal = x + w//2

        if y > half_vertical_total: #Si el obstaculo esta arriba
            message_obs = 'por arriba del obstaculo'
            h = -h
        else:
            message_obs = 'por abajo del obstaculo'

        half_vertical = y + h//2

        # La correccion es un numero entre -1 y 1, donde 0 es el la superposicion de la coordenada del
        # centro de la camara y la coordenada del obstaculo
        if half_horizontal > centerHorizontal:
            correction_horizontal = -(half_horizontal - centerHorizontal) / centerHorizontal
        else:
            correction_horizontal = half_horizontal / centerHorizontal
        if half_vertical > centerVertical:
            correction_vertical = -(half_vertical - centerVertical) / centerVertical
        else:
            correction_vertical = half_vertical / centerVertical

        print(f'correccion: ({correction_horizontal:.2f}%, {correction_vertical:.2f}%)')

        # Rectangulo direccion Zagi y posicionamiento del centro
        cv2.circle(imgContour, (half_horizontal, half_vertical), 10, (0, 0, 255), cv2.FILLED)
        cv2.rectangle(imgContour, (x,y),(x+w,y+h), (0, 0, 255), 2)

        # Move servo motor
        if half_horizontal < centerHorizontal - 30:
            message = 'Mover izquierda, '
        elif half_horizontal > centerHorizontal + 30:
            message = 'Mover derecha, '
        else:
            message = 'Mover recto horizontal, '

        if half_vertical < centerVertical - 30:
            print(message, 'abajo, ', message_obs)
        elif half_vertical > centerVertical + 30:
            print(message, 'arriba, ', message_obs)
        else:
            print(message, 'recto vertical, ', message_obs)
    except:
        #print('Sin contornos')
        pass
#----------------------------------------------------------------BUCLE------------------------------------------------------------------
while True:
    #Frames
    ret, img = cap.read()
    #img = imgobs
    imgContour = img.copy()
    blurred_gaussian = cv2.GaussianBlur(img, (9, 9), 2)
    imgHSV = cv2.cvtColor(blurred_gaussian, cv2.COLOR_BGR2HSV)
    imgAux = np.zeros_like(img)

    # Mascaras
    greenMask = cv2.inRange(imgHSV, lowerGreen, upperGreen)
    pinkMask = cv2.inRange(imgHSV, lowerPink, upperPink)
    mask = cv2.bitwise_or(greenMask, pinkMask)

    # Obtiene los contornos de las mascara clasificados por color
    # getContours(imgContour, mask)

    getContours(imgContour, greenMask, "GREEN")
    getContours(imgContour, pinkMask, "PINK")

    listContour = []
    listHorizontal = []
    listVertical = []
    # Interseccion de mascara e imagen orginal(BGR)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Muestra matriz de imagenes
    imgStack = stackImages(0.4, ([result, imgContour]))
    cv2.imshow('obstaculo verde', imgStack)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()