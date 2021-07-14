import cv2
import numpy as np
import glob
import random

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
# custom object ismi
classes = ["maske"]
# Images path
images_path = glob.glob(r"C:\Users\User\Desktop\mask\test\*.jpg")
# bütün katmanları yazdırıyoruz.
layer_names = net.getLayerNames()
# bütün katmanlar arasından ihtiyacımız olan katmanlarıı alıyoruz.
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# rastgele renk üretiyor label'ımız için
colors = np.random.uniform(0, 255, size=(len(classes), 2))
# Resimleri karıştırarak gösteriyoruz
random.shuffle(images_path)
# tüm görüntüler arasında döngümüz 
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.7, fy=0.6)
    #         resieze(resim,boyut,x kordinatı uzunluğu, y kordinatı uzunluğu)  
    height, width, channels = img.shape

    # nesne tespiti
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # blobFromImage(resim,ölçeklendirme,çerçeve boyut, ortalama , ortalama aktif olması için, kesme)
    
    # Ağ için yeni giriş değerini ayarlar.
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []

    # doğruluk oranlarının tutulduğu dizi
    confidences = []

    # maske olan yüzlerde ekrana bastığımız kutuların tutulduğu dizi
    boxes = []
    for out in outs:
        for detection in out:
            # arama yapıyoruz
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]            
            if confidence > 0.5:
                # Object detected
                #print(class_id)
                # koordinatlar ve yükseklik genişlikleri alıyoruz
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                # dikdörtgen çizimi
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # yükseklik genişlik koordinatları kutunun içine ekliyoruz.
                boxes.append([x, y, w, h])
                # resimlerimizin doğruluk oranları 0 - 1 arası
                confidences.append(float(confidence))
                # class'ımızın id'si dönüyor "0"
                class_ids.append(class_id)

    # NMSBoxes aynı nesnenin birden fazla tespitini engellemek için kullanılıyor
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #                 NMSBoxes(kutu , doğruluk oranları dizisi , minimum tahmin oranı , tahmin eşiği)
    
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # class ismimiz olan "maske" yazımızı alıyoruz
            conf = float("{:2f}".format(confidences[i]))  
            label = str(classes[class_ids[i]]) + str(conf)
            # dikdörtgen ve "maske" yazısı için rastgele üreti
            # lmiş ilgili rengi alıyoruz
            color = colors[class_ids[i]]

            # dikdörtgen kutunun özellikleri
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            #   rectangle(resim,koordinat,boyutlar,renk,kalınlık)

            # yazının özellikleri          
            cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
            #   putText(resim,metin,yazının koordinatları,yazı tipi , yazı boyutu , renk , kalınlık)

    # bir görüntüyü bir pencerede görüntülemek için kullanılır
    cv2.imshow("Tespit Edici v2000", img)
    #   imshow(pencerenin sol üstteki başlığı , resim)

    key = cv2.waitKey(0)
    # herhangi bir tuşa basılana kadar pencerenin ekranda görünmesini sağlıyor.

cv2.destroyAllWindows()

""" 
1 –cv2 nedir ? 
Tüm görüntü işleme işlemlerinde kullanılan ve bu alanın öncüsü bir kütüphanedir. ) açık kaynak kodlu görüntü işleme kütüphanesidir.
2- yolo nedir ?
YOLO (You Only Look Once) bir nesne tanıma yöntemidir. 
3- numpy nedir?
NumPy, Python programlama dili için büyük, çok boyutlu dizileri ve matrisleri destekleyen, bu diziler üzerinde çalışacak üst düzey matematiksel işlevler ekleyen bir kitaplıktır.
4- Glob nedir ?
Glob modülü, Python'da belirli bir klasör içindeki dosyaları listelememize yardımcı olan bir modüldür. Bu modülü kullanırken filtreleme yaparak, sadece istenilen dosyaların listelenmesini de sağlayabiliyoruz.
5- dnn nedir ?
yüz tanıma için derin sinir ağları (deep neural networks-dnn) modülü sunmaktadır. Bu modül TensorFlow derin öğrenme platformlarını desteklemektedir. Bu popüler platformdan eğitilmiş modelleri yükleme ve kullanma imkanı sunan OpenCV ayrıca farklı sınıflandırma modelleri, nesne tespiti, cinsiyet ve yaş tespiti gibi pek çok probleme çözüm sunmaktadır.
"""

