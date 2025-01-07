#version 07.01.2025

import cv2 as cv        #Импорт библиотеки OpenCV
import numpy as np      #Импорт библиотеки Numpy
import sys
from tkinter import Tk  #Импорт библиотеки Tkinter для отображения диалогового окна для выбора изображения
from tkinter.filedialog import askopenfilename

dct = {}        #Словарь образцов изображений
key_size = 3    #Размер ключа в словаре (3х3)
val_size = 6    #Размер значения в словаре (6х6)

#Составление словаря образцов
def dict_build(size, image, i, j, d):
    tpl = [0]*(size * size)
    ind = 0
    for m in range(size):
        for n in range(size):
            tpl[ind] = int(image[d * i + m][ d * j + n])
            ind += 1
    tpl = tuple(tpl)
    return tpl

#Составление словаря для двух масштабов
def dict_build_two_scales(height, width, image1, image2, key_size, val_size):
    for i in range(int(height) - 2):
        for j in range(int(width) - 2):
            #Рассчет ключа словаря 3х3 
            key_3x3 = dict_build(key_size, image2, i, j, 1)
            #Рассчет значения словаря 6х6
            value_6x6 = dict_build(val_size, image1, i, j, 2)
            #Занесение в словарь
            dct[key_3x3] = value_6x6

#Получение ключа по значению в словаре
def get_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key

#Функция нахождения Евклидова расстояния
def euclidean_dist(k1, k2):  
    p1 = np.array(k1)
    p2 = np.array(k2)
    sum_square = np.sum(np.dot(p1 - p2, p1 - p2))
    dist = np.sqrt(sum_square)
    return dist

#Метод ГауссаДОДЕЛАТЬ!!!
def gauss_method(N, A, B):
    n = N                       #Число неизвестных
    a = np.zeros((n, n + 1))    #Инициализация матрицы со значениями соседних образцов в словаре
    #b = np.zeros((n, n + 1))    #Инициализация матрицы со значениями рассматриваемого образца
    x = np.zeros(n)             #Инициализация вектора решений
    cnt = 0

    #Построение матрицы
    for i in range(n):
        for j in range(n + 1):
            if j != n:
                a[i][j] = float(A[cnt])
                cnt += 1
            else:
                a[i][n] = float(B[i])

    # # Applying Gauss Elimination
    # for i in range(n):
    #     if a[i][i] == 0.0:
    #         sys.exit('Divide by zero detected!')
        
    #     for j in range(i+1, n):
    #         ratio = a[j][i]/a[i][i]
        
    #         for k in range(n+1):
    #             a[j][k] = a[j][k] - ratio * a[i][k]

    # # Back Substitution
    # x[n-1] = a[n-1][n]/a[n-1][n-1]

    # for i in range(n-2,-1,-1):
    #     x[i] = a[i][n]
    
    #     for j in range(i+1,n):
    #         x[i] = x[i] - a[i][j]*x[j]
    
    #     x[i] = x[i]/a[i][i]

    # # Displaying solution
    # print('\nRequired solution is: ')
    # for i in range(n):
    #     print('X%d = %0.2f' %(i,x[i]), end = '\t')
    return a

print("Super-Resolution from a single image")
slct = input("Выбрать изображение для обработки? [y - Да/n - Выйти]: ")
if slct == 'n':
    print("Вы вышли. Хорошего дня!")
    sys.exit()

#Выбор файла с изображением
Tk().withdraw()
filename = askopenfilename()

#Если файл не выбран - завершаем программу
while filename == '':
    print('Файл изображения не выбран:(')
    sys.exit()

img = cv.imread(filename, 0)  #0 для отображения черно-белого изображения
        #height = img.shape[0] -- Высота оригинального изображения
        #width = img.shape[1] -- Ширина оригинального изображения
print("Вы выбрали изображение: ", filename)
print("Размер изображения: ", img.shape[1], "x", img.shape[0])

#Уменьшения оригинального изображения в 2, 4 и 8 раз
img2 = cv.resize(img, (int(img.shape[1]  * 0.5), int(img.shape[0]  * 0.5)))
img4 = cv.resize(img2, (int(img2.shape[1] * 0.5), int(img2.shape[0] * 0.5)))
img8 = cv.resize(img4, (int(img4.shape[1] * 0.5), int(img4.shape[0] * 0.5)))

#Результирующее изображение
img_res = cv.resize(img, (int(img.shape[1]  * 2), int(img.shape[0]  * 2)))

print("Сбор словаря...")
#Сбор словаря между изображениями, масштаб которых отличается в 2 раза
dict_build_two_scales(img2.shape[1], img2.shape[0], img, img2, key_size, val_size)
dict_build_two_scales(img4.shape[1], img4.shape[0], img2, img4, key_size, val_size)
dict_build_two_scales(img8.shape[1], img8.shape[0], img4, img8, key_size, val_size)
print("Словарь размером ", len(dct)," значений готов.")

#Проверка пикселей (реализация методом ближайшего соседа)
sorted(dct)
for i in range(img.shape[0] - 2):
    for j in range(img.shape[1] - 2):
        key_res = dict_build(key_size, img, i, j, 1)
        val_res = dct.get(key_res)
        if val_res == None:
            for k, v in dct.items():
                dst = euclidean_dist(key_res, k)
                if dst < 1.5: #Предварительно берем образцы, у которых евклидово расстояние меньше 1.5
                    print(key_res, k, i, j, dst)
                    img_res[2 * i + 2, 2 * j + 2] = v[15]
                    img_res[2 * i + 2, 2 * j + 3] = v[16]
                    img_res[2 * i + 3, 2 * j + 2] = v[21]
                    img_res[2 * i + 3, 2 * j + 3] = v[22]
                    break
        else:
            img_res[2 * i + 2, 2 * j + 2] = val_res[15]
            img_res[2 * i + 2, 2 * j + 3] = val_res[16]
            img_res[2 * i + 3, 2 * j + 2] = val_res[21]
            img_res[2 * i + 3, 2 * j + 3] = val_res[22]

cv.imshow("Result", img_res)
#cv.imshow("Original", img)
cv.waitKey(0)
cv.destroyAllWindows()
print("Обработка изображения закончена!")