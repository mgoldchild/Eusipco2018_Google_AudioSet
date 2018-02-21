"""
Summary:  Audio Set classification for ICASSP 2018 paper
Author:   Qiuqiang Kong, Yong Xu
Created:  2017.11.02

Summary:  Audio Set classification for Eusipco 2018 paper
Author:   Changsong Yu
Modified:  2018.02.21

"""

import numpy as np
import random

first_level_label = [72, 137]
second_level_label = [40, 61, 64, 73, 86, 108, 138, 283, 286, 288, 298, 300, 343, 388, 404, 500, 513]
third_level_label = [0, 8, 14, 15, 16, 22, 25, 26, 27, 37, 38, 39, 41, 47, 49, 50, 51, 52, 53, 54, 55, 
                     56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 70, 71, 74, 81, 87, 90, 93, 95, 97,
                     98, 109, 111, 122, 123, 126, 132, 134, 136, 139, 152, 161, 184, 185, 189, 195, 199,
                     200, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 226, 227, 228, 229, 232,
                     233, 234, 235, 236, 237, 239, 248, 251, 252, 253, 254, 256, 258, 260, 263, 264, 265, 
                     266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 
                     284, 285, 287, 289, 290, 292, 293, 294, 296, 297, 299, 301, 306, 328, 335, 344, 348, 
                     349, 350, 351, 352, 353, 354, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 
                     373, 374, 375, 377, 378, 379, 380, 381, 382, 383, 384, 387, 389, 395, 396, 398, 400, 
                     401, 402, 405, 406, 409, 418, 426, 437, 441, 444, 465, 466, 467, 468, 469, 470, 471,
                     472, 473, 474, 475, 476, 477, 478, 479, 480, 486, 487, 488, 489, 490, 491, 493, 494,
                     495, 496, 497, 498, 499, 501, 504, 505, 506, 507, 508, 509, 510, 511, 512, 519, 520, 
                     521, 522, 523, 524, 525]
fourth_level_label = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 23, 24, 28, 29, 30, 32,
                      33, 34, 35, 36, 42, 43, 44, 45, 46, 48, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 88,
                      89, 91, 92, 94, 96, 99, 102, 104, 106, 110, 112, 115, 116, 117, 119, 121, 124, 125, 
                      127, 128, 129, 131, 133, 135, 140, 147, 148, 149, 150, 151, 153, 155, 158, 160, 162,
                      164, 171, 173, 174, 175, 177, 178, 179, 186, 187, 188, 190, 191, 193, 194, 196, 197,
                      198, 201, 202, 203, 204, 205, 207, 218, 220, 221, 222, 223, 224, 225, 230, 231, 238,
                      240, 241, 242, 243, 244, 245, 246, 247, 249, 250, 255, 257, 259, 261, 262, 291, 295,
                      302, 303, 304, 305, 307, 316, 321, 322, 326, 327, 329, 332, 333, 334, 336, 339, 340,
                      341, 342, 345, 346, 347, 355, 357, 358, 359, 360, 361, 376, 385, 386, 390, 391, 392,
                      393, 394, 397, 399, 403, 407, 408, 410, 411, 412, 413, 414, 415, 416, 419, 420, 421,
                      422, 423, 424, 427, 432, 434, 435, 436, 438, 439, 440, 442, 443, 445, 446, 447, 448, 
                      449, 452, 453, 454, 455, 456, 457, 458, 481, 482, 483, 484, 485, 492, 502, 503, 514, 
                      515, 516, 517, 518, 526]
fifth_level_label = [ 31, 100, 101,103, 105, 107, 113, 114, 118, 120, 130, 141, 142, 143, 144, 145,
                     146, 154, 156, 157, 159, 163, 165, 168, 169, 170, 172, 176, 180, 181, 182, 183, 
                     192, 206, 308, 310, 311, 312, 313, 314, 315, 317, 318, 319, 320, 323, 324, 325, 
                     330, 331, 337, 338, 356, 417, 425, 428, 429, 430, 431, 433, 450, 451, 459, 460,
                     462, 463]
sixth_level_label = [166, 167, 309, 461, 464]



class DataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=None):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        
    def generate(self, xs, ys):
        x = xs[0]
        y = ys[0]
        batch_size = self._batch_size_
        n_samples = len(x)
        
        index = np.arange(n_samples)
        np.random.shuffle(index)
        
        iter = 0
        epoch = 0
        pointer = 0
        while True:
            if (self._type_ == 'test') and (self._te_max_iter_ is not None):
                if iter == self._te_max_iter_:
                    break
            iter += 1
            if pointer >= n_samples:
                epoch += 1
                if (self._type_) == 'test' and (epoch == 1):
                    break
                pointer = 0
                np.random.shuffle(index)                
 
            batch_idx = index[pointer : min(pointer + batch_size, n_samples)]
            pointer += batch_size
            yield x[batch_idx], y[batch_idx]
            
class RatioDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100, verbose=1):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self._verbose_ = verbose
            
    def get_lb_list(self, n_samples_list):
        lb_list = []
        for idx in range(len(n_samples_list)):
            n_samples = n_samples_list[idx]
            lb_list += [idx]
        return lb_list
    def _get_lb_list(self, n_samples_list):
        lb_list = []
        for idx in range(len(n_samples_list)):
            n_samples = n_samples_list[idx]
            lb_list += [idx]
        return lb_list
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        
        if self._verbose_ == 1:
            
            print("n_samples_list: %s" % (n_samples_list,))
            print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        index_list = []
        for i1 in range(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in range(n_labs):
            np.random.shuffle(index_list[i1])
        
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]
            
            for i1 in range(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_per_class_list[i1], len_list[i1])]
                batch_x.append(x[per_class_batch_idx])
                batch_y.append(y[per_class_batch_idx])
                pointer_list[i1] += n_per_class_list[i1]
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

    def random_generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        #id_record = np.zeros((1,id_record))
        id_info = np.arange(n_samples)
        np.random.shuffle(id_info)
        temp_id_info = id_info#np.arange(n_samples)
        #np.random.shuffle(temp_id_info)
        #queue_id = np.arange(batch_size)
        #np.random.shuffle(queue_id)
        print("test_1")
        batch_n = 0
        iter = 0
        while True:
            print("test_2")
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            print("test_3")
            if temp_id_info.shape[0] < batch_size:
                np.random.shuffle(id_info)
                l_n = temp_id_info.shape[0]
                queue_id = np.concatenate((temp_id_info, id_info[:batch_size-l_n]))
                temp_id_info = id_info[batch_size-l_n:]
                for i in range(batch_size):
                    batch_x.append(np.expand_dims(x[queue_id[i]], axis=0))
                    batch_y.append(np.expand_dims(y[queue_id[i]], axis=0))
                    #id_record[0, queue_id[i]] = id_record[0, queue_id[i]] + 1 
            else:
                print("test_4")
                queue_id = temp_id_info[:batch_size]
                temp_id_info = np.delete(temp_id_info, np.arange(batch_size))
                for i in range(batch_size):
                    batch_x.append(np.expand_dims(x[queue_id[i]], axis=0))
                    batch_y.append(np.expand_dims(y[queue_id[i]], axis=0))
                    #id_record[0, queue_id[i]] = id_record[0, queue_id[i]] + 1 
            print("test_5")
            batch_n = batch_n + 1
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            print("test: shape of batch_y:", batch_y.shape)
            print("iteration:", iter)
            yield batch_x, batch_y #id_record

    def new_generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        if self._verbose_ == 1:
            #print("n_samples_list: %s" % (n_samples_list,))
            #print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        sample_training_record = np.zeros((1, n_samples))
        index_list = []
        for j1 in range(n_labs):
            index_list.append(np.where(y[:, j1] == 1)[0])
        for j1 in range(n_labs):
            np.random.shuffle(index_list[j1]) 
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        epoch_num = 1
        _g_labels = sixth_level_label + fifth_level_label + fourth_level_label + third_level_label + second_level_label + first_level_label
        _g_labels_rese = first_level_label + second_level_label + third_level_label + fourth_level_label + fifth_level_label + sixth_level_label
        #_g_labels = g_labels
        #_g_labels_rese = g_labels_rese
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]
            if n_samples - len(np.where(sample_training_record>=epoch_num)[0]) < batch_size: #or iter == 65:
                sample_training_record[np.where(sample_training_record<epoch_num)] = epoch_num
                for j1 in range(n_labs):
                    np.random.shuffle(index_list[j1]) 
                pointer_list = [0] * n_labs
                for j1 in range(n_labs):
                    np.random.shuffle(index_list[j1]) 
                _g_labels = sixth_level_label + fifth_level_label + fourth_level_label + third_level_label + second_level_label + first_level_label
                _g_labels_rese = first_level_label + second_level_label + third_level_label + fourth_level_label + fifth_level_label + sixth_level_label
                epoch_num = epoch_num + 1
            for i1 in _g_labels:
                if pointer_list[i1] >= len_list[i1]:
                    continue
                if n_per_class_list[i1] == 0:
                    continue
                cur_sample_index_record = []
                while True:
                    if len(cur_sample_index_record) >= n_per_class_list[i1]:
                        break
                    if pointer_list[i1] >= len_list[i1]:
                        _g_labels.remove(i1)
                        _g_labels_rese.remove(i1)
                        break
                    per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + 1, len_list[i1])]
                    if len(per_class_batch_idx.tolist()) == 0:
                        break
                    pointer_list[i1] += 1 
                    id_x = per_class_batch_idx.tolist()[0]
                    if sample_training_record[0][id_x] < epoch_num:
                       sample_training_record[0][id_x] += 1
                       cur_sample_index_record.append(id_x)
                    else:
                       continue
                if len(cur_sample_index_record) == 0:
                    continue
                per_class_batch_idx = np.asarray(cur_sample_index_record)
                batch_y.append(y[per_class_batch_idx])
                batch_x.append(x[per_class_batch_idx])
            if len(batch_y) == 0:
                y_count = 0#np.concatenate(batch_y, axis=0).shape[0]
            else:
                y_count = np.concatenate(batch_y, axis=0).shape[0]
            while y_count < batch_size:
                for i1 in _g_labels_rese:
                    if y_count >= batch_size:
                        break
                    cur_sample_index_record = []
                    if pointer_list[i1] >= len_list[i1]:
                       _g_labels.remove(i1)
                       _g_labels_rese.remove(i1)
                       continue
                    per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + 1, len_list[i1])]
                    if len(per_class_batch_idx.tolist()) == 0:
                       continue
                    pointer_list[i1] += 1 
                    id_x = per_class_batch_idx.tolist()[0]
                    if sample_training_record[0][id_x] < epoch_num:
                       sample_training_record[0][id_x] += 1
                       cur_sample_index_record.append(id_x)
                    else:
                       continue
                    if len(cur_sample_index_record) == 0:
                       continue
                    per_class_batch_idx = np.asarray(cur_sample_index_record)
                    batch_y.append(y[per_class_batch_idx])
                    batch_x.append(x[per_class_batch_idx])
                    y_count = y_count + len(cur_sample_index_record)
                if y_count >= batch_size:
                    break
            batch_y = np.concatenate(batch_y, axis=0)
            batch_x = np.concatenate(batch_x, axis=0)
            yield batch_x, batch_y

    def new_generate_1(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        if self._verbose_ == 1:
            #print("n_samples_list: %s" % (n_samples_list,))
            #print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        sample_training_record = np.zeros((1, n_samples))
        index_list = []
        for j1 in range(n_labs):
            index_list.append(np.where(y[:, j1] == 1)[0])
        for j1 in range(n_labs):
            np.random.shuffle(index_list[j1]) 
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        epoch_num = 1
        _g_labels = sixth_level_label + fifth_level_label + fourth_level_label + third_level_label + second_level_label + first_level_label
        _g_labels_rese = first_level_label + second_level_label + third_level_label + fourth_level_label + fifth_level_label + sixth_level_label
        #_g_labels = g_labels
        #_g_labels_rese = g_labels_rese
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]
            if n_samples - len(np.where(sample_training_record>=epoch_num)[0]) < 500000: #or iter == 65:
                sample_training_record[np.where(sample_training_record<epoch_num)] = epoch_num
                for j1 in range(n_labs):
                    np.random.shuffle(index_list[j1]) 
                pointer_list = [0] * n_labs
                for j1 in range(n_labs):
                    np.random.shuffle(index_list[j1]) 
                _g_labels = sixth_level_label + fifth_level_label + fourth_level_label + third_level_label + second_level_label + first_level_label
                _g_labels_rese = first_level_label + second_level_label + third_level_label + fourth_level_label + fifth_level_label + sixth_level_label
                epoch_num = epoch_num + 1
            for i1 in _g_labels:
                if pointer_list[i1] >= len_list[i1]:
                    continue
                if n_per_class_list[i1] == 0:
                    continue
                cur_sample_index_record = []
                while True:
                    if len(cur_sample_index_record) >= n_per_class_list[i1]:
                        break
                    if pointer_list[i1] >= len_list[i1]:
                        _g_labels.remove(i1)
                        _g_labels_rese.remove(i1)
                        break
                    per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + 1, len_list[i1])]
                    if len(per_class_batch_idx.tolist()) == 0:
                        break
                    pointer_list[i1] += 1 
                    id_x = per_class_batch_idx.tolist()[0]
                    if sample_training_record[0][id_x] < epoch_num:
                       sample_training_record[0][id_x] += 1
                       cur_sample_index_record.append(id_x)
                    else:
                       continue
                if len(cur_sample_index_record) == 0:
                    continue
                per_class_batch_idx = np.asarray(cur_sample_index_record)
                batch_y.append(y[per_class_batch_idx])
                batch_x.append(x[per_class_batch_idx])
            if len(batch_y) == 0:
                y_count = 0#np.concatenate(batch_y, axis=0).shape[0]
            else:
                y_count = np.concatenate(batch_y, axis=0).shape[0]
            while y_count < batch_size:
                for i1 in _g_labels_rese:
                    if y_count >= batch_size:
                        break
                    cur_sample_index_record = []
                    if pointer_list[i1] >= len_list[i1]:
                       _g_labels.remove(i1)
                       _g_labels_rese.remove(i1)
                       continue
                    per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + 1, len_list[i1])]
                    if len(per_class_batch_idx.tolist()) == 0:
                       continue
                    pointer_list[i1] += 1 
                    id_x = per_class_batch_idx.tolist()[0]
                    if sample_training_record[0][id_x] < epoch_num:
                       sample_training_record[0][id_x] += 1
                       cur_sample_index_record.append(id_x)
                    else:
                       continue
                    if len(cur_sample_index_record) == 0:
                       continue
                    per_class_batch_idx = np.asarray(cur_sample_index_record)
                    batch_y.append(y[per_class_batch_idx])
                    batch_x.append(x[per_class_batch_idx])
                    y_count = y_count + len(cur_sample_index_record)
                if y_count >= batch_size:
                    break
            batch_y = np.concatenate(batch_y, axis=0)
            batch_x = np.concatenate(batch_x, axis=0)
            yield batch_x, batch_y

    def new_generate_2(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        if self._verbose_ == 1:
            #print("n_samples_list: %s" % (n_samples_list,))
            #print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        sample_training_record = np.zeros((1, n_samples))
        index_list = []
        for j1 in range(n_labs):
            index_list.append(np.where(y[:, j1] == 1)[0])
        for j1 in range(n_labs):
            np.random.shuffle(index_list[j1]) 
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        epoch_num = 1
        _g_labels = sixth_level_label + fifth_level_label + fourth_level_label + third_level_label + second_level_label + first_level_label
        _g_labels_rese = first_level_label + second_level_label + third_level_label + fourth_level_label + fifth_level_label + sixth_level_label
        #_g_labels = g_labels
        #_g_labels_rese = g_labels_rese
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]
            if n_samples - len(np.where(sample_training_record>=epoch_num)[0]) < 1000000: #or iter == 65:
                sample_training_record[np.where(sample_training_record<epoch_num)] = epoch_num
                for j1 in range(n_labs):
                    np.random.shuffle(index_list[j1]) 
                pointer_list = [0] * n_labs
                for j1 in range(n_labs):
                    np.random.shuffle(index_list[j1]) 
                _g_labels = sixth_level_label + fifth_level_label + fourth_level_label + third_level_label + second_level_label + first_level_label
                _g_labels_rese = first_level_label + second_level_label + third_level_label + fourth_level_label + fifth_level_label + sixth_level_label
                epoch_num = epoch_num + 1
            for i1 in _g_labels:
                if pointer_list[i1] >= len_list[i1]:
                    continue
                if n_per_class_list[i1] == 0:
                    continue
                cur_sample_index_record = []
                while True:
                    if len(cur_sample_index_record) >= n_per_class_list[i1]:
                        break
                    if pointer_list[i1] >= len_list[i1]:
                        _g_labels.remove(i1)
                        _g_labels_rese.remove(i1)
                        break
                    per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + 1, len_list[i1])]
                    if len(per_class_batch_idx.tolist()) == 0:
                        break
                    pointer_list[i1] += 1 
                    id_x = per_class_batch_idx.tolist()[0]
                    if sample_training_record[0][id_x] < epoch_num:
                       sample_training_record[0][id_x] += 1
                       cur_sample_index_record.append(id_x)
                    else:
                       continue
                if len(cur_sample_index_record) == 0:
                    continue
                per_class_batch_idx = np.asarray(cur_sample_index_record)
                batch_y.append(y[per_class_batch_idx])
                batch_x.append(x[per_class_batch_idx])
            if len(batch_y) == 0:
                y_count = 0#np.concatenate(batch_y, axis=0).shape[0]
            else:
                y_count = np.concatenate(batch_y, axis=0).shape[0]
            while y_count < batch_size:
                for i1 in _g_labels_rese:
                    if y_count >= batch_size:
                        break
                    cur_sample_index_record = []
                    if pointer_list[i1] >= len_list[i1]:
                       _g_labels.remove(i1)
                       _g_labels_rese.remove(i1)
                       continue
                    per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + 1, len_list[i1])]
                    if len(per_class_batch_idx.tolist()) == 0:
                       continue
                    pointer_list[i1] += 1 
                    id_x = per_class_batch_idx.tolist()[0]
                    if sample_training_record[0][id_x] < epoch_num:
                       sample_training_record[0][id_x] += 1
                       cur_sample_index_record.append(id_x)
                    else:
                       continue
                    if len(cur_sample_index_record) == 0:
                       continue
                    per_class_batch_idx = np.asarray(cur_sample_index_record)
                    batch_y.append(y[per_class_batch_idx])
                    batch_x.append(x[per_class_batch_idx])
                    y_count = y_count + len(cur_sample_index_record)
                if y_count >= batch_size:
                    break
            batch_y = np.concatenate(batch_y, axis=0)
            batch_x = np.concatenate(batch_x, axis=0)
            yield batch_x, batch_y

