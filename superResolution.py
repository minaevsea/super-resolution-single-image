#version 23.12.2024

import cv2 as cv        #Импорт библиотеки OpenCV
import numpy as np      #Импорт библиотеки Numpy
import sys

dct = {}        #Словарь образцов изображений
key_size = 3    #Размер ключа в словаре (3х3)
val_size = 6    #Размер значения в словаре (6х6)

#Составление словаря образцов
def dict_build(size, image, i, j, d):
    tpl = [0]*(size * size)
    ind = 0
    for m in range(size):
        for n in range(size):
            tpl[ind] = image[d * i + m, d * j + n]
            ind += 1
    tpl = tuple(tpl)
    return tpl

#Получение ключа по значению в словаре
def get_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key

#Функция нахождения Евклидова расстояния
def euclidean_dist(k1, k2):  
    p1 = np.array(k1)
    p2 = np.array(k2)
    dist = np.sqrt(np.dot(p1 - p2, p1 - p2))
    return dist

#Метод Гаусса (НЕ ЗАКОНЧЕНО!!!)
def gauss_method(N, A, B):
    n = N                       #Число неизвестных
    a = np.zeros((n, n + 1))    #Инициализация матрицы со значениями соседних образцов в словаре
    x = np.zeros(n)             #Инициализация вектора решений
    cnt = 0

    #Построение матрицы
    for i in range(n):
        for j in range(n + 1):
            if j != n:
                print("A: ", A[cnt])
                a[i][j] = float(A[cnt])
                cnt += 1
            else:
                print("B: ", B[i])
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

print("\nSuper-Resolution from a single image")

print("Image openning...")
img = cv.imread("lena.jpg", 0)  #0 для отображения черно-белого изображения
height = img.shape[0]           #Высота оригинального изображения
width = img.shape[1]            #Ширина оригинального изображения

img2 = cv.resize(img, (int(width * 1/2), int(height * 1/2)))    #Уменьшение оригинального изображения в 2 раза
img4 = cv.resize(img, (int(width * 1/4), int(height * 1/4)))    #Уменьшение оригинального изображения в 4 раза
img8 = cv.resize(img, (int(width * 1/8), int(height * 1/8)))    #Уменьшение оригинального изображения в 8 раз

print("Dictionary building...")
for i in range(int(height * 1/2) - 2):
    for j in range(int(width * 1/2) - 2):
        #Рассчет ключа словаря 3х3 
        # key_3x3 = [0]*(key_size * key_size)
        # ind333 = 0
        # for m in range(key_size):
        #     for n in range(key_size):
        #         key_3x3[ind333] = img2[i + m, j + n]
        #         ind333 += 1
        # key_3x3 = tuple(key_3x3)
        key_3x3 = dict_build(key_size, img2, i, j, 1)

        #Рассчет значения словаря 6х6
        # value_6x6 = [0]*(val_size * val_size)
        # ind = 0
        # for m in range(val_size):
        #     for n in range(val_size):
        #         value_6x6[ind] = img[2 * i + m, 2 * j + n]
        #         ind += 1
        # value_6x6 = tuple(value_6x6)
        value_6x6 = dict_build(val_size, img, i, j, 2)

        dct[key_3x3] = value_6x6    #Занесение в словарь

print("key_3x3 ", key_3x3)
print("value_6x6 ", value_6x6)

dct['123'] = '123'
arr2 = tuple(np.multiply(key_3x3, 2)) #Перевод в кортеж
print(len(dct))

ddd = euclidean_dist((1, 2, 3), (4, 5, 6))
print(ddd)
kv = get_key(dct, 'key_6x6_2')
print(kv)
gm = gauss_method(3, key_3x3, arr2)
print(gm)

cv.imshow("LENA", img)
cv.waitKey(0)
cv.destroyAllWindows()
print("Image was closed!")