import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
from sklearn.decomposition import PCA

from detection.utils import image_utils
from detection.utils.image_utils import get_annotated_img, filename_to_id
from detection.utils.logger import logger
from detection.utils.time_utils import time_it

np.random.seed(43)

gaussian_k = image_utils.gaussian_kernel((31, 31), 8, 1)


class FrameInfo():
    '''
    Contains annotated image information such as image data, annotations etc.
    '''

    def __init__(self, base_dir, img_id, roi, annotations, bbox=(15, 15)):
        self.base_dir = base_dir
        self.img_id = img_id
        # add buffer to region of interest ...
        self.roi = roi[0] - bbox[0], roi[1] + bbox[0], roi[2] - bbox[1], roi[3] + bbox[1]
        self.bbox = bbox
        self.full_img = np.array(cv2.imread(os.path.join(self.base_dir, self.img_id)), dtype=np.float32)
        self.img_data = self.full_img[self.roi[2]: self.roi[3], self.roi[0]: self.roi[1]]
        # normalize annotations for cropped image...
        self.annotations = [(ann[0] - self.roi[0], ann[1] - self.roi[2], ann[2]) for ann in annotations]
        self.all_seq_patches = []

    def annotated_img(self):
        annotated_img = get_annotated_img(self.img_data, self.annotations, self.bbox)
        return annotated_img

    def get_random_patches(self, patch_size, no_of_patches):
        '''
        Randomly samples no_of_patches of patch_size from image.
        '''
        img_shape = self.img_data.shape
        x = np.random.randint(0, img_shape[0] - patch_size[0], no_of_patches)
        y = np.random.randint(0, img_shape[1] - patch_size[1], no_of_patches)
        xy = zip(x, y)
        img_patches = []
        for i, j in xy:
            img_patch = Patch(self, j, i, patch_size)
            img_patches.append(img_patch)
        return img_patches

    def sequential_patches(self, patch_size, step_size):
        '''
        Returns all sequential patches from image separated by step.
        '''
        if len(self.all_seq_patches) == 0:
            img_shape = self.img_data.shape
            x = range(0, img_shape[0] - patch_size[0], step_size[0])
            y = range(0, img_shape[1] - patch_size[1], step_size[1])
            xy = [(i, j) for i in x for j in y]
            img_patches = []
            for i, j in xy:
                img_patch = Patch(self, j, i, patch_size)
                img_patches.append(img_patch)
            self.all_seq_patches = img_patches
        return self.all_seq_patches


class Patch(object):
    '''
    Represents a patch inside an input image.
    '''

    def __init__(self, frame_info, startx, starty, patch_size):
        self.frame_info = frame_info
        self.startx = startx
        self.starty = starty
        self.patch_size = patch_size
        self.__find_annotations()

    def get_img(self):
        img_data = self.frame_info.img_data
        img_patch = img_data[self.starty:self.starty + self.patch_size[0],
                    self.startx:self.startx + self.patch_size[1]]
        return img_patch

    def __find_annotations(self):
        '''
        Finds annotations whose bounding box completely lie in the patch.
        '''
        annotations = []
        for ann in self.frame_info.annotations:
            x, y, s = ann
            bbox_size = self.frame_info.bbox
            minx, miny, maxx, maxy = x - bbox_size[0], y - bbox_size[1], x + bbox_size[0], y + bbox_size[1]
            if minx >= self.startx and maxx <= self.startx + self.patch_size[1] and miny >= \
                    self.starty and maxy <= self.starty + self.patch_size[1]:
                # if self.startx <= x <= self.startx + self.patch_size[1] and self.starty <= y <= self.starty + \
                #         self.patch_size[1]:
                annotations.append(ann)
        self.ann_relative = annotations
        self.annotations = [(ann[0] - self.startx, ann[1] - self.starty, ann[2]) for ann in annotations]

    def annotated_img(self):
        ann_patch = get_annotated_img(self.get_img(), self.annotations, self.frame_info.bbox)
        return ann_patch

    def ann_mask(self, no_classes):
        img_mask = np.zeros(self.patch_size + (no_classes,))
        for ann in self.annotations:
            x, y, s = ann
            bbox = self.frame_info.bbox
            i = s if no_classes > 1 else 0
            if bbox[0] < x < self.patch_size[0] - bbox[0] and bbox[1] < y < self.patch_size[1] - bbox[1]:
                img_mask[y - bbox[1]:y + bbox[1] + 1, x - bbox[0]:x + bbox[0] + 1, i] = np.maximum(
                    img_mask[y - bbox[1]:y + bbox[1] + 1, x - bbox[0]:x + bbox[0] + 1, i], gaussian_k)
        return img_mask


class ImageDataset(object):
    '''
    Maintains training dataset and implements generators which provide data while training.

    '''

    def __init__(self, base_dir, file_suffix, annotation_file='localizations.txt', normalize=False):
        self.base_dir = base_dir
        self.file_suffix = file_suffix
        self.annotation_file = os.path.join(base_dir, annotation_file)
        self.all_frames = self.load_image_data()
        self.dataset_size = len(self.all_frames)
        if normalize:
            self.channel_mean = self.calc_channel_mean()
            self.normalize_frames()

    def get_frames(self):
        return self.all_frames

    def get_dataset_size(self):
        return self.dataset_size

    def load_image_data(self):
        '''
        Reads the annotation file and create frame objects for all image frames.
        '''
        logger.info('Loading data from directory:{}'.format(self.base_dir))
        all_annotations = {}
        with open(self.annotation_file, 'r') as fr:
            for line in fr:
                frame, x, y, s, _ = list(map(int, line.split()))
                if frame not in all_annotations:
                    all_annotations[frame] = []

                all_annotations[frame].append((x, y, s))

        all_files = os.listdir(self.base_dir)
        all_files = [fn for fn in all_files if fn.endswith(self.file_suffix)]
        all_files.sort(key=lambda x: filename_to_id(x))
        # since image files are listed sequentially in annotation file

        roi = self.get_global_bounds(all_annotations)

        frame_infos = []
        total_annotations = 0
        for i, fn in enumerate(all_files):
            annotations = all_annotations[i] if i in all_annotations else []
            frame_info = FrameInfo(self.base_dir, fn, roi, annotations);
            frame_infos.append(frame_info)
            total_annotations += len(annotations)

        logger.info('Total frames loaded:{}, total annotations:{}', len(frame_infos), total_annotations)
        return frame_infos

    def get_global_bounds(self, all_annotations):
        '''
        Returns largest image region such that it covers complete annotated region in all input images.
        '''
        img_bounds = []
        for key, annotations in all_annotations.items():
            minx = min(annotations, key=lambda ann: ann[0])[0]
            maxx = max(annotations, key=lambda ann: ann[0])[0]
            miny = min(annotations, key=lambda ann: ann[1])[1]
            maxy = max(annotations, key=lambda ann: ann[1])[1]
            img_bounds.append((minx, maxx, miny, maxy))
        gminx = min(img_bounds, key=lambda x: x[0])[0]
        gmaxx = max(img_bounds, key=lambda x: x[1])[1]
        gminy = min(img_bounds, key=lambda x: x[2])[2]
        gmaxy = max(img_bounds, key=lambda x: x[3])[3]

        return gminx, gmaxx, gminy, gmaxy

    def per_pixel_mean(self):
        img_frames = [frame.img_data for frame in self.all_frames]
        return np.mean(img_frames, axis=0)

    def calc_channel_mean(self):
        c1_frames = [frame.img_data[:, :, 0] for frame in self.all_frames]
        c2_frames = [frame.img_data[:, :, 1] for frame in self.all_frames]
        c3_frames = [frame.img_data[:, :, 2] for frame in self.all_frames]
        channel_mean = np.mean(c1_frames), np.mean(c2_frames), np.mean(c3_frames)
        logger.info('Channel mean:{}', channel_mean)
        return channel_mean

    def normalize_frames(self):
        for frame in self.all_frames:
            frame.img_data -= frame.img_data.mean()/ (frame.img_data.std()+1e-8)
        logger.info('Normalized frames with channel mean')

    def save_all_annotated(self, out_dir):
        for frame in self.all_frames:
            ann_frame = frame.img_data
            out_filename = os.path.join(out_dir, frame.img_id)
            cv2.imwrite(out_filename, ann_frame)

    @time_it
    def pixel_pca(self):
        pixel_vectors = []
        for frame in self.all_frames:
            pixel_vector = np.reshape(frame.img_data,
                                      [frame.img_data.shape[0] * frame.img_data.shape[1], frame.img_data.shape[2]])
            pixel_vectors.extend(pixel_vector.tolist())
        print 'Total vectors', len(pixel_vectors)
        pca = PCA(n_components=3)
        pca.fit(np.array(pixel_vectors))
        cov = pca.get_covariance()
        print 'Cov', cov
        w, v  = np.linalg.eig(cov)
        print w,v


if __name__ == '__main__':
    # frame = cv2.imread('/data/lrz/hm-cell-tracking/annotations/in/cam0_0026.jpg')
    # pixel_vector = np.reshape(frame, [frame.shape[0] * frame.shape[1], frame.shape[2]])
    # pca = PCA(n_components=3)
    # out = pca.fit(np.array(pixel_vector))
    # cov = out.get_covariance()
    # print np.linalg.eig(cov)
    dataset_gen = ImageDataset('/data/lrz/hm-cell-tracking/sequences_150602_3T3/sample_01', '.jpg', normalize=True)

    # dataset_gen.pixel_pca()
    # patches = dataset_gen.all_frames[0].sequential_patches((224, 224), (200, 200))
    # plt.figure(0), plt.imshow(patches[0].get_img())
    # plt.figure(1), plt.imshow(patches[0].annotated_img())
    # plt.figure(2), plt.imshow(np.squeeze(patches[0].ann_mask(1)))
    #
    # plt.show()
    # dataset_gen.save_all_annotated('/data/cell_detection/mean_imgs')
    # frame = dataset_gen.all_frames[1]
    # patches = frame.sequential_patches((224, 224), (200,200))
    # patch = patches[0]
    # mask = patch.binary_mask()
    # plt.figure(1)
    # plt.imshow(patch.ann_patch())
    # plt.figure(2)
    # plt.imshow(np.squeeze(mask))
    # plt.show()
    # avg_img = dataset_gen.per_pixel_mean()
    # frame = dataset_gen.all_frames[0]
    # patches = frame.get_random_patches((224, 224), 10)
    # for batch in dataset_gen.fcn_data_generator(5, (224, 224), 2):
    #     for i in range(len(batch[0]['input_1'])):
    #         patch_mask = batch[1]['class_out'][i]
    #         print(patch_mask.shape)
    #         plt.figure(1)
    #         plt.imshow(batch[0]['input_1'][i])
    #         for j in range(1):
    #             plt.figure(j + 2)
    #             plt.imshow(patch_mask[:, :, j], interpolation='nearest')
    #         plt.show()

    # for frame in dataset_gen.all_frames:
    #     mask =
    # image_utils.draw_prediction()
    # print(len(dataset_gen.all_frames))
    # for frame in dataset_gen.all_frames:
    #     cv2.imwrite('/data/out1/{}'.format(frame.img_id), frame.annotated_img())
    #
    # print dataset_gen.calc_channel_mean()
    # patch = frame.sequential_patches((224, 224), (200, 200))[0]
    # out_img = model.predict(patch)
    # ann_img = image_utils.draw_prediction(out_img)
    # cv2.imwrite('/home/sanjeev/out.png', ann_img)
    # cv2.imwrite('/home/sanjeev/avg_img.png', avg_img)
    # for x in dataset_gen.patch_dataset_generator(1):
    #     print x
    #     break
    # for frame in dataset_gen.all_frames:
    #     if frame.img_id == '00402_bw.png':
    #         break
    # patches = frame.sequential_patches((64, 64), (60, 60))

    # for data in dataset_gen.patch_dataset_generator(100):
    #     for i in range(100):
    #         ann_img = image_utils.draw_prediction(data[0]['input_1'][i], data[1]['class_out'][i], data[1]['bb_out'][i])
    #         cv2.imwrite('/data/predic_{}.png'.format(i), ann_img)
    #     print data
    #     break
    #
    # for data in dataset_gen.grid_patch_dataset_generator(200, patch_size=(224, 224), grid_size=(32, 32), nb_objects=5):
    #     img_data = data[0]['input_1']
    #     labels, bboxes = data[1]['class_out'], data[1]['bb_out']
    # print img_data.shape, labels.shape, bboxes.shape
    # for i in range(len(img_data)):
    #     img_ann = image_utils.draw_prediction(patch.get_img(), labels, bboxes)
    # cv2.imwrite('/data/patches/pred{}.png'.format(i), img_ann)
    # label_map, bbox_map = dataset_gen.grid_ground_truth(patches[0], (28, 28))
    # print label_map.shape, bbox_map.shape
    # annotations = image_utils.feature_to_annotations(patches[0].get_img(), label_map, bbox_map)
    # annotated_img = np.array(patches[0].get_img())
    # for ann in annotations:
    #     x, y, minx, miny, maxx, maxy = ann
    #     cv2.rectangle(annotated_img, (minx, miny), (maxx, maxy), (0, 0, 255), 2)
    # cv2.imwrite('/data/ann.png', annotated_img)
    # print len(patches[0].annotations)
