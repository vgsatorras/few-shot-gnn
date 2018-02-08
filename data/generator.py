from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import random
from torch.autograd import Variable
from . import omniglot
from . import mini_imagenet


class Generator(data.Dataset):
    def __init__(self, root, args, partition='train', dataset='omniglot'):
        self.root = root
        self.partition = partition  # training set or test set
        self.args = args

        assert (dataset == 'omniglot' or
                dataset == 'mini_imagenet'), 'Incorrect dataset partition'
        self.dataset = dataset

        if self.dataset == 'omniglot':
            self.input_channels = 1
            self.size = (28, 28)
        else:
            self.input_channels = 3
            self.size = (84, 84)

        if dataset == 'omniglot':
            self.loader = omniglot.Omniglot(self.root, dataset=dataset)
            self.data = self.loader.load_dataset(self.partition == 'train', self.size)
        elif dataset == 'mini_imagenet':
            self.loader = mini_imagenet.MiniImagenet(self.root)
            self.data, self.label_encoder = self.loader.load_dataset(self.partition, self.size)
        else:
            raise NotImplementedError

        self.class_encoder = {}
        for id_key, key in enumerate(self.data):
            self.class_encoder[key] = id_key

    def rotate_image(self, image, times):
        rotated_image = np.zeros(image.shape)
        for channel in range(image.shape[0]):
            rotated_image[channel, :, :] = np.rot90(image[channel, :, :], k=times)
        return rotated_image

    def get_task_batch(self, batch_size=5, n_way=20, num_shots=1, unlabeled_extra=0, cuda=False, variable=False):
        # Init variables
        batch_x = np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32')
        labels_x = np.zeros((batch_size, n_way), dtype='float32')
        labels_x_global = np.zeros(batch_size, dtype='int64')
        target_distances = np.zeros((batch_size, n_way * num_shots), dtype='float32')
        hidden_labels = np.zeros((batch_size, n_way * num_shots + 1), dtype='float32')
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way*num_shots):
            batches_xi.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
        # Iterate over tasks for the same batch

        for batch_counter in range(batch_size):
            positive_class = random.randint(0, n_way - 1)

            # Sample random classes for this TASK
            classes_ = list(self.data.keys())
            sampled_classes = random.sample(classes_, n_way)
            indexes_perm = np.random.permutation(n_way * num_shots)

            counter = 0
            for class_counter, class_ in enumerate(sampled_classes):
                if class_counter == positive_class:
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_], num_shots+1)
                    # Test sample is loaded
                    batch_x[batch_counter, :, :, :] = samples[0]
                    labels_x[batch_counter, class_counter] = 1
                    labels_x_global[batch_counter] = self.class_encoder[class_]
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_], num_shots)

                for s_i in range(0, len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :, :, :] = samples[s_i]
                    if s_i < unlabeled_extra:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 0
                        hidden_labels[batch_counter, indexes_perm[counter] + 1] = 1
                    else:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(positive_class)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi,
                      torch.from_numpy(hidden_labels)]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def cast_variable(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_variable(input[i])
        else:
            return Variable(input)

        return input
