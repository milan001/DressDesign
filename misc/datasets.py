from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import random
from misc.config import cfg


class Dataset(object):
    def __init__(self, imageIds, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._imageIds = np.array(imageIds)
        self._images=[]
        self._embeddings = np.array(embeddings)
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_batch_in_file=cfg.NUM_BATCH_IN_FILE
        self._num_examples = 15*self._num_batch_in_file*64 #len(imageIds)
        self._saveIDs = self.saveIDs()
        self._fake_images = []
        self._fake_file_id = 0

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._File_index = -1
        self._batch_index_in_file = self._num_batch_in_file
        self._num_files = 91
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None

    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def saveIDs(self):
        self._saveIDs = np.arange(self._num_examples)
        np.random.shuffle(self._saveIDs)
        return self._saveIDs

    def readCaptions(self, filenames, class_id):
        name = filenames
        if name.find('jpg/') != -1:  # flowers dataset
            class_name = 'class_%05d/' % class_id
            name = name.replace('jpg/', class_name)
        cap_path = '%s/text_c10/%s.txt' %\
                   (self.workdir, name)
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def transform(self, images):
        if self._aug_flag:
            transformed_images =\
                np.zeros([images.shape[0], self._imsize, self._imsize, 3])
            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                h1 = np.floor((ori_size - self._imsize) * np.random.random())
                w1 = np.floor((ori_size - self._imsize) * np.random.random())
                cropped_image =\
                    images[i][w1: w1 + self._imsize, h1: h1 + self._imsize, :]
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(cropped_image)
                else:
                    transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images

    def sample_embeddings(self, embeddings, filenames, class_id, sample_num):
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []
            for i in range(batch_size):
                randix = np.random.choice(embedding_num,
                                          sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    captions = self.readCaptions(filenames[i],
                                                 class_id[i])
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def next_batch(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        #print("New batch..."+str(self._index_in_epoch)+" "+str(self._batch_index_in_file)+" "+str(self._File_index))
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        self._batch_index_in_file += 1
        

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            self._File_index = 0
            self._batch_index_in_file = self._num_batch_in_file + 1
            # Shuffle the data
            start = 0
        
            # Start next epoch
            assert batch_size <= self._num_examples
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        
        if self._batch_index_in_file >= self._num_batch_in_file:
            self._File_index += 1
            self._batch_index_in_file = 0
            with open(cfg.DATASET_NAME+"/76images" + str(self._File_index-1) + ".pickle") as f:
                self._images=np.array(pickle.load(f))
            self._fake_file_id=np.random.randint(self._num_files-1)
            if self._fake_file_id > self._File_index:
                self._fake_file_id += 1
            with open(cfg.DATASET_NAME+"/76images" + str(self._fake_file_id) + ".pickle") as f:
                self._fake_images=np.array(pickle.load(f))
            self._perm = np.arange(self._num_batch_in_file * batch_size)
            np.random.shuffle(self._perm)
        
        start_file = self._batch_index_in_file * batch_size
        end_file = start_file + batch_size
        

        current_ids = self._perm[start_file:end_file]
        fake_ids = np.random.randint(self._fake_images.shape[0], size=batch_size)
        
        sampled_images = self._images[current_ids]
        sampled_wrong_images = self._fake_images[fake_ids]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.

        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [sampled_images, sampled_wrong_images]

        
        if self._embeddings is not None:
            '''filenames = [self._filenames[i] for i in current_ids]
            class_id = [self._class_id[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[self._imageIds[(self._File_index-1)*self._num_batch_in_file * batch_size+current_ids[i] for i in range(len(current_ids))]],
                                       filenames, class_id, window)
            ret_list.append(sampled_embeddings)
            ret_list.append(sampled_captions)  '''
            Id=((self._File_index-1)*self._num_batch_in_file*batch_size+current_ids)
            ret_list.append(self._embeddings[self._imageIds[Id]])
        else:
            ret_list.append(None)
            ret_list.append(None)

        ''' if self._labels is not None:
            ret_list.append(self._labels[[current_ids]])
        else:
            ret_list.append(None) '''
            
        return ret_list

    def next_batch_test(self, batch_size, start, max_captions):
        """Return the next `batch_size` examples from this data set."""
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            start = end - batch_size
        else:
            end = start + batch_size

        sampled_images = self._images[start:end]
        sampled_images = sampled_images.astype(np.float32)
        # from [0, 255] to [-1.0, 1.0]
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_images = self.transform(sampled_images)

        sampled_embeddings = self._embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []

        sampled_captions = []
        sampled_filenames = self._filenames[start:end]
        sampled_class_id = self._class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i],
                                         sampled_class_id[i])
            # print(captions)
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(np.squeeze(batch))

        return [sampled_images, sampled_embeddings_batchs,
                self._saveIDs[start:end], sampled_captions]


class TextDataset(object):
    def __init__(self, workdir, embedding_type, hr_lr_ratio):
        lr_imsize = cfg.TEST.LR_IMSIZE
        self.hr_lr_ratio = hr_lr_ratio
        if self.hr_lr_ratio == 1:
            self.image_filename = '/File_Ids.pickle'
        elif self.hr_lr_ratio == 4:
            self.image_filename = '/File_Ids.pickle'

        self.image_shape = [lr_imsize * self.hr_lr_ratio,
                            lr_imsize * self.hr_lr_ratio, 3]
        self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        self.embedding_shape = None
        self.train = None
        self.test = None
        self.workdir = workdir
        if embedding_type == 'cnn-rnn':
            self.embedding_filename = '/list_attr_img.txt'
        elif embedding_type == 'skip-thought':
            self.embedding_filename = '/list_attr_img.txt'

    def get_data(self, pickle_path, aug_flag=True):
        with open(pickle_path + self.image_filename, 'rb') as f:
            imageIds = pickle.load(f)
        f=pickle_path + self.embedding_filename
#        embeddings=np.loadtxt(f,skiprows=2,usecols=range(1,1001),dtype='int8')
#            embeddings = pickle.load(f)
#        embeddings = np.array(embeddings)
        embeddings = np.empty([289222,1000],dtype='int8')
        self.embedding_shape = [embeddings.shape[-1]]
        print('embeddings: ', embeddings.shape)
        with open(pickle_path + '/filenames.txt', 'rb') as f:
            list_filenames = f.readlines()
            print('list_filenames: ', len(list_filenames), list_filenames[0])
#        with open(pickle_path + '/class_info.pickle', 'rb') as f:
#            class_id = pickle.load(f)
        print("Check...")

        return Dataset(imageIds, self.image_shape[0], embeddings,
                       list_filenames, self.workdir, None,
                       aug_flag)#, class_id)



# next(f)
# next(f)
# embeddings=[]
# name=[]
# for line in f:
#     temp=line.split();
#     name.append(temp[0])
#     embeddings.append([int(x) for x in temp[1:]])
