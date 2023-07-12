from paddle.io import Dataset
import os
import cv2 
import numpy as np 


class Dataset_GAN(Dataset):
    #
    def __init__(self, path_image):
        self.path_image = path_image
        self.list_image = os.listdir(path_image)
    #
    def __getitem__(self, idx):
        image = cv2.imread(self.path_image + self.list_image[idx]) / 127.5 -1
        # image = np.array(Image.open(self.path_image + self.list_image[idx])) / 127.5 -1
        # h_random = np.random.randint(20, 198-128)
        # w_random = np.random.randint(0, 178-128)
        # image = image[h_random:h_random+128, w_random:w_random+128, :]
        image = image[20:198, :, :]
        image = cv2.resize(image, (64, 64))
        if np.random.randint(2):
            image = image[:, ::-1, :]
        image = image[:, :, ::-1]
        image = image.transpose([2,0,1])
        return image.astype('float32')
    #
    def __len__(self):
        return len(self.list_image)