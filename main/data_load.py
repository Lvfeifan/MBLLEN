from glob import glob
import numpy as np
import random
import scipy
import os
import cv2 as cv

class Dataloader():
    def __init__(self, dataset_name, crop_shape=(256, 256)):
        self.dataset_name = dataset_name
        self.crop_shape = crop_shape

    def imread_color(self, path):
        img = cv.imread(path, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH)/255.
        b, g, r = cv.split(img)
        img_rgb = cv.merge([r, g, b])
        return img_rgb

    def imwrite(self, path, img):
        r, g, b = cv.split(img)
        img_rgb = cv.merge([b, g, r])
        cv.imwrite(path, img_rgb)

    def load_data(self, batch_size=16):
        path = glob('../dataset/train/*.jpg')
        self.n_batches = int(len(path) / batch_size)
        while 1:
            random.shuffle(path)
            for i in range(self.n_batches - 1):
                batch_path = path[i * batch_size:(i + 1) * batch_size]
                input_imgs = np.empty((batch_size, self.crop_shape[0], self.crop_shape[1], 6), dtype="float32")
                gt = np.empty((batch_size, self.crop_shape[0], self.crop_shape[1], 3), dtype="float32")

                number = 0
                for img_B_path in batch_path:
                    img_B = self.imread_color(img_B_path)
                    path_mid = os.path.split(img_B_path)
                    path_A_1 = path_mid[0] + '_' + self.dataset_name
                    path_A = os.path.join(path_A_1, path_mid[1])
                    img_A = self.imread_color(path_A)

                    nw = random.randint(0, img_B.shape[0] - self.crop_shape[0])
                    nh = random.randint(0, img_B.shape[1] - self.crop_shape[1])

                    crop_img_A = img_A[nw:nw + self.crop_shape[0], nh:nh + self.crop_shape[1], :]
                    crop_img_B = img_B[nw:nw + self.crop_shape[0], nh:nh + self.crop_shape[1], :]

                    if np.random.randint(2, size=1)[0] == 1:  # random flip
                        crop_img_A = np.flipud(crop_img_A)
                        crop_img_B = np.flipud(crop_img_B)
                    if np.random.randint(2, size=1)[0] == 1:
                        crop_img_A = np.fliplr(crop_img_A)
                        crop_img_B = np.fliplr(crop_img_B)
                    if np.random.randint(2, size=1)[0] == 1:  # random transpose
                        crop_img_A = np.transpose(crop_img_A, (1, 0, 2))
                        crop_img_B = np.transpose(crop_img_B, (1, 0, 2))

                    input_imgs[number, :, :, :] = np.concatenate([crop_img_A, crop_img_B], axis=-1)
                    gt[number, :, :, :] = crop_img_B
                    number += 1
                yield input_imgs, gt
