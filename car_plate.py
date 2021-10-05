import cv2 # pip install opencv-python
import imutils # pip install imutils
import numpy as np # pip install numpy
import pytesseract # pip install pytesseract
import PIL # pip install Pillow
from PIL import ImageTk
from PIL import Image
import tkinter as tk #python yüklenirken indiriyor



def goruntu_tara():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # https://tesseract-ocr.github.io/tessdoc/Downloads.html
    #https://digi.bib.uni-mannheim.de/tesseract/
    img = cv2.imread('unnamed.jpg',cv2.IMREAD_COLOR)
    # img = cv2.resize(img, (400,300) )
    # Eğer sabit bir yerde plaka okuyacak ise daha optimize
    #olması için resmin kırpılması

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resmi alıp gri çeviriyor
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    #(source_image, diameter of pixel, sigmaColor, sigmaSpace)

    cany = cv2.Canny(gray, 30, 200) # 30, 200 yoğunluk min ve max
    #cany edge kenar algılama algoritması
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
    #https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/

    contours = cv2.findContours(cany.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #kontür alma
    #https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html

    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    #https://stackoverflow.com/questions/55062579/opencv-how-to-clear-contour-from-noise-and-false-positives
    screenCnt = 0
    #(cnts, contours, heirarchy) = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #python eski sürümünde 3 değer alabiliyor

    #cnts = contours[0]
    #screenCnt = 0

    #for contour in contours:
    #    [x,y,w,h] = cv2.boundingRect(contour)
    #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
    for c in contours: # -> https://stackoverflow.com/questions/62274412/cv2-approxpolydp-cv2-arclength-how-these-works

        _ = cv2.arcLength(c, True)
        #https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c
        approx = cv2.approxPolyDP(c, 0.018 * _, True)
        #daha düzgün dikdörtgen algılatmak
        # dörtgen bir cisim
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
    else:
        detected = 1 # eğer varsa

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3) # kontür çizmek
    # https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html bu linkte örneklendirme yapılmış

    mask = np.zeros(gray.shape,np.uint8)
    #shape = satır sütun piksel sayısı ve renkli ise renk katmanları
    #np.uint8 = Unsigned integer (0 to 255)
    #np.zeros = np.zeros() bir tuple (demet) değeri alır.
    #Bu tuple değeri, oluşturmak istediğimiz dizinin boyutlarının değerleridir.

    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    #https://stackoverflow.com/questions/44333605/what-does-bitwise-and-operator-exactly-do-in-opencv/52429616
    #algılanan resmin dışındakileri siyah yapar


    (x, y) = np.where(mask == 255)
    #np.where(mask<5,-1,100) şeklinde çalışsaydı eğer
    #koşulu sağlamadığında -1 değerini atayacak sağladığında 100
    #np.where(mask==255) olduğunda ise true false döndürecek

    (tx, ty) = (np.min(x), np.min(y)) # dizinin minimum ve maksimum elemanını buluyor
    (bx, by) = (np.max(x), np.max(y))
    Cropped = gray[tx:bx+1, ty:by+1] #tx = bx + 1 kadar, ty=by +1 kadar
    custom_config = r'-l eng --psm 6' #config l dil kodu --psm
    text = pytesseract.image_to_string(Cropped, config=custom_config) #psm 11
    # yazıyı okuma
    #-> çeşitli yazı okuma örnekleri https://github.com/NanoNets/ocr-with-tesseract/blob/master/tesseract-tutorial.ipynb
    #-> açıklamalı örnekler https://nanonets.com/blog/ocr-with-tesseract/

    print("Araç Plakası: ",text)
    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(200,70))
    cv2.imshow('car',img)
    cv2.imshow('Cropped',Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite("carkaydet.jpg",img)
    #cv2.imwrite("carneli.jpg",Cropped)


# def form():
#   form = tk.Tk() # form açılması
#   form.title("CinGöz")
#   form.geometry("1400x750")
#   resim1 = ImageTk.PhotoImage(Image.open("carkaydet.jpg")) #resimleri çekme
#   resim2 = ImageTk.PhotoImage(Image.open("carneli.jpg"))
#   label1 = tk.Label(form,image=resim1) #
#   label1.pack(side=tk.TOP)
#   label2 = tk.Label(form, image=resim2)
#   label2.place(x=1330,y=450)
#   form.mainloop()


goruntu_tara()

#form()

