# Exp 5 Implementation-of-filter
### Date: 15-03-2024
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:

### Step 1:
Import necessary libraries: OpenCV, NumPy, and Matplotlib.Read an image, convert it to RGB format, define an 11x11 averaging kernel, and apply 2D convolution filtering.Display the original and filtered images side by side using Matplotlib.

### Step 2:
Define a weighted averaging kernel (kernel2) and apply 2D convolution filtering to the RGB image (image2).Display the resulting filtered image (image4) titled 'Weighted Averaging Filtered' using Matplotlib's imshow function.

### Step 3:
Apply Gaussian blur with a kernel size of 11x11 and standard deviation of 0 to the RGB image (image2).Display the resulting Gaussian-blurred image (gaussian_blur) titled 'Gaussian Blurring Filtered' using Matplotlib's imshow function.

### Step 4:
Apply median blur with a kernel size of 11x11 to the RGB image (image2).Display the resulting median-blurred image (median) titled 'Median Blurring Filtered' using Matplotlib's imshow function.

### Step 5 :
Define a Laplacian kernel (kernel3) and perform 2D convolution filtering on the RGB image (image2).Display the resulting filtered image (image5) titled 'Laplacian Kernel' using Matplotlib's imshow function.

### Step 6 :
Apply the Laplacian operator to the RGB image (image2) using OpenCV's cv2.Laplacian function.Display the resulting image (new_image) titled 'Laplacian Operator' using Matplotlib's imshow function.

## Program:
```
Developed By : NATHIN R
Register Number : 212222230090
```
### 1. Smoothing Filters

i) Using Averaging Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('dog.png')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel = np.ones((11,11), np. float32)/121
image3 = cv2.filter2D(image2, -1, kernel)

plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title('Orignal')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')
```
![image](https://github.com/Augustine0306/Implementation-of-filter/assets/119404460/5de29806-ecb1-4efc-8748-a1e565cd59a8)

ii) Using Weighted Averaging Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('dog.png')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image4 = cv2.filter2D(image2, -1, kernel2)
plt.imshow(image4)
plt.title('Weighted Averaging Filtered')
```
![image](https://github.com/Augustine0306/Implementation-of-filter/assets/119404460/9708fb2a-f8c7-4474-8e3f-69ff60982952)



iii) Using Gaussian Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('dog.png')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

gaussian_blur = cv2.GaussianBlur(src=image2, ksize=(11,11), sigmaX=0, sigmaY=0)
plt.imshow(gaussian_blur)
plt.title(' Gaussian Blurring Filtered')
```

![image](https://github.com/Augustine0306/Implementation-of-filter/assets/119404460/af71280e-756e-4377-be46-ce30ac12ef1c)

iv) Using Median Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('dog.png')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

median=cv2.medianBlur (src=image2, ksize=11)
plt.imshow(median)
plt.title(' Median Blurring Filtered')
```
![image](https://github.com/Augustine0306/Implementation-of-filter/assets/119404460/7f93d043-6e49-4ff8-9a4c-310dacce8b1e)


### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('dog.png')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel3 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
image5 =cv2.filter2D(image2, -1, kernel3)
plt.imshow(image5)
plt.title('Laplacian Kernel')
```
![image](https://github.com/Augustine0306/Implementation-of-filter/assets/119404460/16608642-b7d0-487f-a37d-6ae6bbee78f8)


ii) Using Laplacian Operator
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('dog.png')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

new_image = cv2.Laplacian (image2, cv2.CV_64F)
plt.imshow(new_image)
plt.title('Laplacian Operator')
```
![image](https://github.com/Augustine0306/Implementation-of-filter/assets/119404460/1d90e713-f2c6-4341-ad4f-7565113b633e)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
