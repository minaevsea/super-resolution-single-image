import cv2 as cv

print("Super-Resolution from a single image")

print("Image openning...")
img = cv.imread("lena.jpg")
cv.imshow("LENA", img)
cv.waitKey(0)
cv.destroyAllWindows()
print("Image was closed!")

print("Hi to GitHub!!!")