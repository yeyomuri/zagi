# Antes de la competencia, ajustar los rangos de colores HSV con ayuda de los controles deslizables (sliders)
# para obtener una máscara nítida en función de los colores del entorno que capture la camara del Zagi.

import cv2
import numpy as np
from stack_images import stackImages

# Rangos HSV
lowerGreen = np.array([40, 72, 59])
upperGreen = np.array([77, 255, 255])
lowerPink = np.array([150, 84, 180])
upperPink = np.array([179, 255, 255])
cap = cv2.VideoCapture(0)

# Asigna resolusion de la camara
cap.set(3, 480)
cap.set(4, 320)
_, frame = cap.read()
row, cols, _ = frame.shape
centerHorizontal = cols // 2
centerVertical = row // 2

# ----- Función que clasifica y ordena los contornos por color, tambien hace el cálculo por donde tiene que pasar el Zagi -----
def getContours(imgContour, img, label):

    listContour = []
    listHorizontal = []
    listVertical = []
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area>300:
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
        # Variables de mensajes
        message = ''
        message_obs = ''

        # Variables de correccion
        correction_horizontal = 0
        correction_vertical = 0

        #Coordenadas. El simbolo ° representan los puntos del obstaculo con coordenadas (x1,y1), (x2,y2) y (0, y3) 

        #   °(x1,y1)
        #   ||         ||
        #   ||         ||
        #   || (0, y3) ||
        #   ||----°----||
        #   ||---------||
        #   ||         ||
        #              ° (x2,y2)

        x1 = listVertical[0][1]+ listVertical[0][3]
        x2 = listVertical[-1][1] 
        y1 = listVertical[0][2]
        y2 = listVertical[-1][2] + listVertical[-1][4]
        
        x_center = x1 + abs(x2 - x1) // 2
        y_center = y1 + abs(y2 - y1) // 2

        if len(listHorizontal) == 0:
            # Rectangulo direccion Zagi y posicionamiento del centro
            cv2.circle(imgContour, (x_center, y_center), 10, (0, 0, 255), cv2.FILLED)
            cv2.rectangle(imgContour, (x1,y1),(x2,y2), (0, 0, 255), 2)

        else:
            y3 = listHorizontal[0][2] + (listHorizontal[0][4] // 2)
            y_up_center = y1 + abs(y3 - y1) // 2
            y_down_center = y2 - abs(y2 - y3) // 2

            if(y3 > y_center): #Si el obstaculo esta abajo                
                cv2.circle(imgContour, (x_center, y_up_center), 10, (0, 0, 255), cv2.FILLED)
                cv2.rectangle(imgContour, (x1,y1),(x2,y3), (0, 0, 255), 2)

                #Correccion vertical
                if y_up_center > centerVertical:
                    correction_vertical = - abs(y_up_center - centerVertical) / centerVertical
                else:
                    correction_vertical = 1 - (y_up_center / centerVertical)
            
            elif(y3 < y_center): #Si el obstaculo esta arriba
                cv2.circle(imgContour, (x_center, y_down_center), 10, (0, 0, 255), cv2.FILLED)
                cv2.rectangle(imgContour, (x1,y3),(x2,y2), (0, 0, 255), 2)
                
                #Correccion vertical
                if y_down_center > centerVertical:
                    correction_vertical = - abs(y_down_center - centerVertical) / centerVertical
                else:
                    correction_vertical = 1 - (y_down_center / centerVertical)
                
            # La correccion es un numero entre -1 y 1, donde 0 es el la superposicion de la coordenada del
            # centro de la camara y la coordenada del centro del obstaculo
            if x_center > centerHorizontal:
                correction_horizontal = - abs(x_center - centerHorizontal) / centerHorizontal
            else:
                correction_horizontal = 1 - (x_center / centerHorizontal)


            print(f'correccion: ({correction_horizontal:.2f}%, {correction_vertical:.2f}%)')

            ##
            #   A partir de la CORRECION se crea un controlador PID para el seguimiento de la coordenada
            #   del obstaculo.
            #

    except:
        # Sin contornos
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
    mask = cv2.bitwise_or(greenMask, pinkMask) #Une las dos mascaras (verde y rosada)

    # Obtiene los contornos de las mascara clasificados por color
    # getContours(imgContour, mask)
    getContours(imgContour, mask, "PINK")
    #getContours(imgContour, greenMask, "GREEN")
    

    # Interseccion de mascara e imagen orginal(BGR)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Muestra matriz de imagenes
    imgStack = stackImages(0.4, ([result, imgContour]))
    cv2.imshow('obstaculo verde', imgStack)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
