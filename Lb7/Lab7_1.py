import cv2  
import numpy as np
import matplotlib.pyplot as plt

# 1 Поиск шаблона на изображении

# Загружаем изображение
rgb_img = cv2.imread('17_jpg.jpg') 
plt.figure()
plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

# Преобразуем изображение в оттенки серого 
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
  
# Загружаем шаблон  
template = cv2.imread('temp2.jpg')
plt.figure()
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))

# Преобразуем в оттенки серого
gray_templ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  


# Считаем размеры шаблона
w, h = template.shape[:-1] 
 
# Вызываем функцию cv2.matchTemplate для вычисления метрики схожести
# в качестве параметров передаем изображение, шаблон и тип вычисляемой метрики
res = cv2.matchTemplate(gray_img,gray_templ,cv2.TM_CCOEFF_NORMED)  

# Возможные варианты метрик:
#    cv2.TM_SQDIFF — сумма квадратов разниц значений пикселей
#    cv2.TM_SQDIFF_NORMED — сумма квадрат разниц цветов, отнормированная в диапазон 0..1.
#    cv2.TM_CCORR — сумма поэлементных произведений шаблона и сегмента картинки
#    cv2.TM_CCORR_NORMED — сумма поэлементных произведений, отнормированное в диапазон -1..1.
#    cv2.TM_CCOEFF — кросс-коррелация изображений без среднего
#    cv2.TM_CCOEFF_NORMED — кросс-корреляция между изображениями без среднего, отнормированная в -1..1 (корреляция Пирсона)
plt.figure()
plt.imshow(res, cmap='jet')
plt.colorbar()


# Определяем порог для выделения области локализации шаблона на изобажении
# Порог зависит от метрики, т.к. значения различных метрик могут различаться
# на порядки. Кроме по своей сути некоторые метрики измеряют "похожесть" 
# и имеют большие значения для похожих изображений, а другие измеряют "отличие",
# и ноборот, большие значения появляются для различающихся изображений
threshold = 0.8

# Определяем точки изображения в которых метрика превышает порог
# Эти точки - центры локализации шаблона
# Знак сравнения для метрик, измеряющих "отличия" необходимо заменить на противоположный
loc = np.where(res >= threshold)  

# Вокруг выделенных максимумов обводим прямоугольники с размерами шаблона
plot_img = rgb_img.copy()
for pt in zip(*loc[::-1]):
    cv2.rectangle(plot_img, pt,(pt[0] + w, pt[1] + h),(0,255,255), 8)  

# Отображаем результат на графике
plt.figure()
plt.imshow(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))

# Если необходимо учесть разность в масштабах шаблона и изображения,
# строится пирамида изображений, на элементах коотрой выполняется поиск шаблона

# Пример построения пирамиды изображений
layer = rgb_img.copy() 
for i in range(3):
# Функция pyrDown() уменьшает масштаб изображения, а  pyrUp() - увеличивает
# По умолчанию масштаб изменяется в два раза
    layer = cv2.pyrDown(layer)
    cv2.imshow("str(i)", layer)
    cv2.waitKey(0)


# 2 Для снижения влияния яркости изображения или шаблона на результат, 
# рекомендуется делать поиск не по исходным изображениям, а по выделенным
# из них границам

# Границы вычисляем на изображении в оттенках серого
imgG = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

# Для выделения границ используем фильтр Собеля в двух направлениях
x = cv2.Sobel(imgG,cv2.CV_16S,1,0)  
y = cv2.Sobel(imgG,cv2.CV_16S,0,1)  

# берем модуль от результата применения фильтра Собеля
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)  
  
# объединяем "вертикальные" и "горизонтальные" границы в одно изображение
dstI = cv2.addWeighted(absX,0.5,absY,0.5,0)

plt.subplot(2,1,1),plt.imshow(imgG,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(dstI,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])


# Такие же преобразования делаем и с шаблоном, но изменяем яркость шаблона
tmpG = 2*cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(tmpG,cv2.CV_16S,1,0)  
y = cv2.Sobel(tmpG,cv2.CV_16S,0,1)  
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)  
dstT = cv2.addWeighted(absX,0.5,absY,0.5,0)

plt.subplot(3,1,1),plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2),plt.imshow(tmpG,cmap = 'gray')
plt.title('Changed'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3),plt.imshow(dstT,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

# Считаем размеры шаблона
w, h = tmpG.shape
 
# Вычисляем метрику схожести  
res = cv2.matchTemplate(dstI,dstT,cv2.TM_CCOEFF_NORMED)  
plt.figure()
plt.imshow(res, cmap='jet')
plt.colorbar()

#  Отбираем максимумы и строим результат на графике
threshold = 0.8
loc = np.where(res >= threshold)  
plot_img = rgb_img.copy()
for pt in zip(*loc[::-1]):
    cv2.rectangle(plot_img, pt,(pt[0] + w, pt[1] + h),(0,255,255), 8)  

plt.figure()
plt.imshow(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))



# 3 Если предполагается что шаблон может отличаться по яркости, масштабу или
# ориентации, то рекомендуется использовать методы выделения особых точек, такие 
# как SIFT, SURF, ORB и т.д.
# такие методы являются более устойчивыми изменениям шаблона или изображения
import cv2  
import numpy as np
import matplotlib.pyplot as plt

# Загружаем изображения
rgb_img = cv2.imread('17_jpg.jpg') 
template = cv2.imread('temp2.jpg')

# Преобразуем в оттенки серого
img1 = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

# Преобразуем и вносим небольшие изменения в шаблон
scale = 1.1 # масштаб изменения размеров
scBr = 0.9 # коэффициент изменения яркости

img2 = cv2.resize(np.uint8(0.9*cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)),
           (int(template.shape[1]*scale), int(template.shape[0]*scale)), interpolation = cv2.INTER_AREA)

# Создаем детектор особых точек
sift = cv2.SIFT_create()
# sift = cv2.xfeatures2d.SIFT_create() # В зависимости от версии opencv может работать эта команда

# Запускаем детектор на изображении и на шаблоне
# Метод возвращает список особых точек и их дескрипторов
k_1, des_1 = sift.detectAndCompute(img1, None)
k_2, des_2 = sift.detectAndCompute(img2, None)

# Каждая особая точка имеет несколько параметров, таких как координаты, 
# размер, угол ориентации, мощность отклика и размер области особой точки.
print(k_1[1].pt)
print(k_1[1].size)
print(k_1[1].angle)
print(k_1[1].response)
print(k_1[1].size)

# Отрисуем найденные точки на картинке
img = cv2.drawKeypoints(img1, k_1, des_1, (0, 255, 255))
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Отрисуем найденные точки на шаблоне
img = cv2.drawKeypoints(img2, k_2, des_2, (0, 255, 255))
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Найденные точки на шаблоне и изображении нужно сопоставить друг с другом
# по их совпадению мы поймем какая часть изображения соотносится с шаблоном
# для этого исмользуется объект BFMatcher, который в качестве параметров принимает
# метрику. По умолчанию используется метрика - cv2.NORM_L2 - среднеквадратичное
# расстояние или cv2.NORM_L1 - модуль разницы координат.
# Её можно использовать для детекторов SIFT и SURF. 
# Для бинарных дескрипторов (в детекторах ORB, BRIEF, BRISK) используется
# расстояние Хэмминга cv2.NORM_HAMMING. 
# Если ORB использует VTA_K == 3 или 4, следует использовать cv2.NORM_HAMMING2.
# Параметр crossCheck указывает что необходимо вернуть ровно одно совпадение для 
# каждой особой точки
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# Дескрипторы используются для сопоставления особых точек шаблона и изображения между 
# собой.
# На выходе получаем кортеж элементов DMatch
# Каждый элемент содержит:
# queryIdx - индекс особой точки на изображении, 
# trainIdx - индекс особой точки шаблона, соответствующего этому совпадению,
# distance -  расстояние между парой совпадающих особых точек, 
# чем меньше это значение, тем ближе находятся две точки. 
matches = bf.match(des_1, des_2)

print(matches[1].queryIdx)
print(matches[1].trainIdx)
print(matches[1].distance)


# Через индексы, содержащиеся в DMatch, можно обращаться к массиву особых точек
# Подставим индекс точки изображения из первого совпадения в массив  
# ключевых точек и выведем ее размер
print(k_1[matches[1].queryIdx].size)

# Для того, чтобы отобрать наилучшие совпадения (пары совпадающих точек)
#  отсортируем кортеж совпадений по расстоянию
matches = sorted(matches, key=lambda x: x.distance)

# и построим эти совпадения на изображении
#  количество совпадающих точек, котовые построятся на изображении указывается
#  в matches[:10]
img3 = cv2.drawMatches(img1, k_1, img2, k_2, matches[:20], img2, flags=2)
plt.figure()
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))


# Для большей устойчивости определения соответствия точек
# может использоваться knnMatch, с параметром k, который определяет не только
# ближайшее совпадение для каждой точки шаблона, но и k ближайших
# Теперь кортеж DMatch будет многомерным (двумерным, если k=2),
# так как каждой точке шаблона соответствует несколько точек изображения
bf = cv2.BFMatcher(cv2.NORM_L1)
matches = bf.knnMatch(des_1, des_2, k=2)

# Лучшие пары особых точек отбираются с использованием теста отношения правдоподобия
good = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good.append([m])

# построим совпадения на изображении
img3 = cv2.drawMatchesKnn(img1,k_1,img2,k_2,good[:200],None,flags=2)
plt.figure()
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))


# Для большей устойчивости определения соответствия точек
# может использоваться knnMatch, с параметром k, который определяет не только
# ближайшее совпадение для каждой точки шаблона, но и k ближайших
# Теперь кортеж DMatch будет многомерным (двумерным, если k=2),
# так как каждой точке шаблона соответствует несколько точек изображения
bf = cv2.BFMatcher(cv2.NORM_L1)
matches = bf.knnMatch(des_1, des_2, k=2)

# Лучшие пары особых точек отбираются с использованием теста отношения правдоподобия
good = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good.append([m])

# построим совпадения на изображении
img3 = cv2.drawMatchesKnn(img1,k_1,img2,k_2,good[:200],None,flags=2)
plt.figure()
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

img = cv2.imread('17_jpg.jpg')
template = cv2.imread('temp2.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_orig, w_orig = template.shape[:2]

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2-x1+1)*(y2-y1+1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]
        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        overlap = (w*h)/area[idxs]
        idxs = idxs[overlap<=overlapThresh]
    return boxes[pick].astype(int)


scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
all_boxes = []
threshold = 0.6

for scale in scales:
    resized_template = cv2.resize(template, (0,0), fx=scale, fy=scale)
    t_h, t_w = resized_template.shape[:2]
    template_gray = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        all_boxes.append([pt[0], pt[1], pt[0]+t_w, pt[1]+t_h])

boxes = non_max_suppression(all_boxes, overlapThresh=0.3)

out = img.copy()
clusters = []
for (x1,y1,x2,y2) in boxes:
    cx = (x1+x2)//2
    cy = (y1+y2)//2
    clusters.append((cx, cy))
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.circle(out, (cx, cy), 10, (255,0,0), 2)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print("Найдено объектов:", len(clusters))







