#!/usr/bin/env python
# coding: utf-8

# In[254]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn import cluster
from sklearn.cluster import KMeans


# In[255]:


def rgb2grayscale(im):
       """
           Converts RGB to Grayscale.
           @params: Input RGB image.
           @return: Grayscale image(1 channel)
       """
       if  len(im.shape) > 2:
           if im.shape[2] == 3: # Convert RGB image to Grayscale
               r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
               grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
               return grayscale
       else:
           return im


# In[262]:


#Step1 : convert original image to grayscale
im = Image.open("sunset.jpg")
im_array=np.array(im)
#img = rgb2grayscale(np.array(im))
#img=np.float32(img)
#print(im_array)


# In[263]:



#Step 2: apply guassian filter
gaussian = gaussian_filter(im_array,7.0)


# In[264]:


#plt.imshow(im_array)


# In[265]:


#plt.imshow(gaussian)


# In[266]:


#Step 3: apply K means clustering algorithm
print(gaussian.shape)

#Step 4 reshape to give 2D array
pixels=gaussian.reshape(gaussian.shape[0]*gaussian.shape[1],gaussian.shape[2])
#print(pixels.shape)


# In[261]:


#Step 4: Kmeans

model = cluster.KMeans(n_clusters=8)
model.fit(X)


# In[267]:


#Step6: Labels array is shaped as a vector and need to be reshaped as an image (width x height)
pixel_centroids=model.labels_
cluster_centers=model.cluster_centers_
pixel_centroids.shape[0]



# In[240]:


final=np.zeros((pixel_centroids.shape[0],3))
final.shape          


# In[268]:


for cluster_no in range(8):
    final[pixel_centroids==cluster_no]=cluster_centers[cluster_no]

final[0:5]                                            


# In[244]:


comp_image=final.reshape(im_array.shape[0],im_array.shape[1],3)
comp_image.shape


# In[269]:


comp_image=Image.fromarray(np.uint8(comp_image))
comp_image.save('compressed.jpg')


# In[ ]:




