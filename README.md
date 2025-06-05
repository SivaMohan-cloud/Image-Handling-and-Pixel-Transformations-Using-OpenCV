# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** SIVAMOHANASUNDARAM V
- **Register Number:** 212222230145

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread('Eagle_in_Flight.jpg',cv2.IMREAD_GRAYSCALE)
```

#### 2. Print the image width, height & Channel.
```python
print(img_gray.shape[1],img_gray.shape[0],1)
```

#### 3. Display the image using matplotlib imshow().
```python
plt.imshow(img_gray,cmap='gray')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```python
cv2.imwrite('Eagle_in_Flight.png',img_gray)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```python
img_color = cv2.imread('Eagle_in_Flight.jpg')
img_color = cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```python
plt.imshow(img_color)
plt.show()
print(img_color.shape[1],img_color.shape[0],img_color.shape[2])
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```python
cropped = img_color[50:500,200:600]
plt.imshow(cropped)
```

#### 8. Resize the image up by a factor of 2x.
```python
resized = cv2.resize(cropped,(0,0),fx=2,fy=2)
plt.imshow(resized)
```

#### 9. Flip the cropped/resized image horizontally.
```python
flipped = cv2.flip(resized,1)
plt.imshow(flipped)
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```python
img_apo = cv2.imread('Apollo-11-launch.jpg')
plt.imshow(img_apo)
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
# YOUR CODE HERE: use putText()
cv2.putText(img_apo,'Apollo 11 Saturn',(50,img_apo.shape[0]-50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
plt.imshow(img_apo)
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rect_color = magenta
# YOUR CODE HERE
cv2.rectangle(img_apo,(400,50),(800,700),(255,0,0),3)
plt.imshow(img_apo)
```

#### 13. Display the final annotated image.
```python
plt.imshow(cv2.cvtColor(img_apo,cv2.COLOR_BGR2RGB))
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```python
boy_img = cv2.imread('boy.jpg')
plt.imshow(boy_img)
```

#### 15. Adjust the brightness of the image.
```python
# Create a matrix of ones (with data type float64)
# matrix_ones = 
# YOUR CODE HERE
matrix_ones = np.ones(boy_img.shape,dtype='uint8')*50
```

#### 16. Create brighter and darker images.
```python
img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)
# YOUR CODE HERE
img_bright = cv2.add(boy_img,matrix_ones)
img_dark = cv2.subtract(boy_img,matrix_ones)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```python
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_dark, img_bright]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

#### 18. Modify the image contrast.
```python
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
matrix1 = 
matrix2 = 
# img_higher1 = 
# img_higher2 = 
# YOUR CODE HERE
matrix1 = np.ones(boy_img.shape, dtype='uint8') * 25
matrix2 = np.ones(boy_img.shape, dtype='uint8') * 50
img_higher1 = cv2.addWeighted(boy_img, 1.1, matrix1, 0, 0)
img_higher2 = cv2.addWeighted(boy_img, 1.2, matrix2, 0, 0)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```python
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_higher1, img_higher2]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```python
b, g, r = cv2.split(boy_img)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([b, g, r]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```python
merged = cv2.merge([b, g, r])
plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
plt.show()

```

#### 22. Split the image into the H, S, V components & Display the channels.
```python
hsv = cv2.cvtColor(boy_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([h, s, v]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```python
merged_hsv = cv2.merge([h, s, v])
plt.imshow(cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB))
plt.show()
```

## Output:
- **i)** Read and Display an Image.
    ![image](https://github.com/user-attachments/assets/657a919c-3edb-4c57-a2f1-c026a0458dba)
    ![image](https://github.com/user-attachments/assets/b1520702-fcf1-46a8-88f5-3fa799379c25)

- **ii)** Adjust Image Brightness.
    ![image](https://github.com/user-attachments/assets/709efa81-14ab-44b1-a178-60c7b16415bf)
  
- **iii)** Modify Image Contrast.
    ![image](https://github.com/user-attachments/assets/881f9917-50eb-496c-b5c5-a632d599fd89)
  
- **iv)** Generate Third Image Using Bitwise Operations.
    ![image](https://github.com/user-attachments/assets/7002a62a-5984-49bd-a0fd-981ed4cce398)


## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

