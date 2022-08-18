import abc
import random
import collections
import numpy as np
import os

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])

"""A general Interface"""


class LabelLoader:
    @abc.abstractmethod
    def get_label(self, sample_filename: str) -> str:
        '''
        return sample label based on filename
        '''
        pass

class YZDDataset:
    def __init__(self,
                 nm_dir,
                 am_dir,
                 sample_hash_list,
                 num_epoch,
                 inital_edge_feature_dim,
                 max_num_node_of_one_graph,
                 max_num_edge_of_one_graph=None,
                 step_per_epoch=None):
        '''
        :param nm_dir: nm is short for node matrix
        :param am_dir: am is short for adjacent matric
        :param sample_hash_list: sample hash is used as sample filename for de-duplication
        :param max_num_node_of_one_graph: for excluding too large graphs
        '''
        self.nm_dir = nm_dir
        self.am_dir = am_dir
        self.num_epoch = num_epoch
        self.inital_edge_feature_dim = inital_edge_feature_dim
        self.max_num_node_of_one_graph = max_num_node_of_one_graph
        self.max_num_edge_of_one_graph = max_num_edge_of_one_graph
        self.inital_nm = np.ones(shape=(max_num_node_of_one_graph + 1, 1))
        self.step_per_epoch = step_per_epoch
        self.label_to_sample_hash_dict = {}
        self.label_list = []
        self.sample_hash_to_label_dict = {}
        self.label_loader = LabelLoader()

        for sample_hash in sample_hash_list:
            samplepath_nm = os.path.join(self.nm_dir, sample_hash + '.nm')
            samplepath_am = os.path.join(self.am_dir, sample_hash + '.am')
            # just our naming convention

            if not os.path.isfile(samplepath_nm) or os.stat(samplepath_am).st_size == 0:
                continue
            sample_label = self.label_loader.get_label(sample_filename=sample_hash)
            self.sample_hash_to_label_dict[sample_hash] = sample_label
            if not sample_label in self.label_to_sample_hash_dict:
                self.label_list.append(sample_label)
                self.label_to_sample_hash_dict[sample_label] = []
            self.label_to_sample_hash_dict[sample_label].append(sample_hash)
        self.num_sample = len(self.sample_hash_to_label_dict)

    def sim_pair_sampling(self):
        sampled_label = random.choice(seq=self.label_list)
        nm_1 = self.inital_nm
        nm_2 = self.inital_nm
        while nm_1.shape[0] >= self.max_num_node_of_one_graph or nm_2.shape[0] >= self.max_num_node_of_one_graph:
            sample_hash_1, sample_hash_2 = random.choices(population=self.label_to_sample_hash_dict[sampled_label], k=2)
            nm_path_1 = os.path.join(self.nm_dir, sample_hash_1 + '.nm')
            nm_path_2 = os.path.join(self.nm_dir, sample_hash_2 + '.nm')
            am_path_1 = os.path.join(self.am_dir, sample_hash_1 + '.am')
            am_path_2 = os.path.join(self.am_dir, sample_hash_2 + '.am')
            try:
                nm_1 = np.loadtxt(nm_path_1, delimiter=',')
            except ValueError:
                continue
            try:
                nm_2 = np.loadtxt(nm_path_2, delimiter=',')
            except ValueError:
                continue
            am_1 = np.loadtxt(am_path_1, delimiter=',').astype(int)
            am_2 = np.loadtxt(am_path_2, delimiter=',').astype(int)
            if self.max_num_edge_of_one_graph != None and (
                    am_1.shape[0] >= self.max_num_edge_of_one_graph or am_2.shape[
                0] >= self.max_num_edge_of_one_graph):
                nm_1 = self.inital_nm
                nm_2 = self.inital_nm
            if len(am_1.shape) == 1:
                am_1 = am_1[None, :]
            if len(am_2.shape) == 1:
                am_2 = am_2[None, :]
        return nm_1, am_1, sample_hash_1, nm_2, am_2, sample_hash_2, 1

    def diff_pair_sampling(self):
        sampled_label_1, sampled_label_2 = random.choices(population=self.label_list, k=2)
        nm_1 = self.inital_nm
        nm_2 = self.inital_nm
        while nm_1.shape[0] >= self.max_num_node_of_one_graph:
            sample_hash_1 = random.choice(seq=self.label_to_sample_hash_dict[sampled_label_1])
            nm_path_1 = os.path.join(self.nm_dir, sample_hash_1 + '.nm')
            am_path_1 = os.path.join(self.am_dir, sample_hash_1 + '.am')
            #
            try:
                nm_1 = np.loadtxt(nm_path_1, delimiter=',')
            except ValueError:
                continue
            am_1 = np.loadtxt(am_path_1, delimiter=',').astype(int)
            if self.max_num_edge_of_one_graph != None and am_1.shape[0] >= self.max_num_edge_of_one_graph:
                nm_1 = self.inital_nm
            if len(am_1.shape) == 1:
                am_1 = am_1[None, :]
        while nm_2.shape[0] >= self.max_num_node_of_one_graph:
            sample_hash_2 = random.choice(seq=self.label_to_sample_hash_dict[sampled_label_2])
            nm_path_2 = os.path.join(self.nm_dir, sample_hash_2 + '.nm')
            am_path_2 = os.path.join(self.am_dir, sample_hash_2 + '.am')
            try:
                nm_2 = np.loadtxt(nm_path_2, delimiter=',')
            except ValueError:
                continue
            am_2 = np.loadtxt(am_path_2, delimiter=',').astype(int)
            if self.max_num_edge_of_one_graph != None and am_2.shape[0] >= self.max_num_edge_of_one_graph:
                nm_1 = self.inital_nm
            if len(am_2.shape) == 1:
                am_2 = am_2[None, :]
        # return nm_1, am_1, nm_2, am_2, -1
        return nm_1, am_1, sample_hash_1, nm_2, am_2, sample_hash_2, -1

    def _pair_generator(self):
        while True:
            random_num = random.uniform(0, 1)
            if random_num < 0.5:
                yield self.sim_pair_sampling()
            else:
                yield self.diff_pair_sampling()

    def _triplet_generator(self):
        while True:
            sampled_label_1, sampled_label_2 = random.choices(population=self.label_list, k=2)
            nm_1 = self.inital_nm
            nm_2 = self.inital_nm
            nm_3 = self.inital_nm
            while nm_1.shape[0] >= self.max_num_node_of_one_graph or nm_2.shape[0] >= self.max_num_node_of_one_graph:
                sample_hash_1, sample_hash_2 = random.choices(population=self.label_to_sample_hash_dict[sampled_label_1],
                                                            k=2)
                nm_path_1 = os.path.join(self.nm_dir, sample_hash_1 + '.nm')
                am_path_1 = os.path.join(self.am_dir, sample_hash_1 + '.am')
                am_path_2 = os.path.join(self.am_dir, sample_hash_2 + '.am')
                nm_path_2 = os.path.join(self.nm_dir, sample_hash_2 + '.nm')
                try:
                    nm_1 = np.loadtxt(nm_path_1, delimiter=',')
                except ValueError:
                    continue
                am_1 = np.loadtxt(am_path_1, delimiter=',').astype(int)
                try:
                    nm_2 = np.loadtxt(nm_path_2, delimiter=',')
                except ValueError:
                    continue
                am_2 = np.loadtxt(am_path_2, delimiter=',').astype(int)
                if self.max_num_edge_of_one_graph != None and (
                        am_1.shape[0] >= self.max_num_edge_of_one_graph or am_2.shape[
                    0] >= self.max_num_edge_of_one_graph):
                    nm_1 = self.inital_nm
                    nm_2 = self.inital_nm
                if len(am_1.shape) == 1:
                    am_1 = am_1[None, :]
                if len(am_2.shape) == 1:
                    am_2 = am_2[None, :]

            while nm_3.shape[0] >= self.max_num_node_of_one_graph:
                sample_hash_3 = random.choice(seq=self.label_to_sample_hash_dict[sampled_label_2])
                nm_path_3 = os.path.join(self.nm_dir, sample_hash_3 + '.nm')
                am_path_3 = os.path.join(self.am_dir, sample_hash_3 + '.am')
                try:
                    nm_3 = np.loadtxt(nm_path_3, delimiter=',')
                except ValueError:
                    continue
                am_3 = np.loadtxt(am_path_3, delimiter=',').astype(int)
                if self.max_num_edge_of_one_graph != None and am_3.shape[0] >= self.max_num_Edge_of_one_graph:
                    nm_3 = self.inital_nm
                if len(am_3.shape) == 1:
                    am_3 = am_3[None, :]

            yield nm_1, am_1, nm_2, am_2, nm_3, am_3

    def pairs(self, batch_size):
        pair_generator = self._pair_generator()
        num_batch_in_total = self.num_epoch * self.step_per_epoch
        for batch_idx in range(num_batch_in_total):
            nm_of_one_batch = []
            am_of_one_batch = []
            label_of_one_batch = []
            sample_hash_of_one_batch = []
            for sample_idx in range(batch_size):
                # nm_1, am_1, nm_2, am_2, label = next(pair_generator)
                nm_1, am_1, sample_hash_1, nm_2, am_2, sample_hash_2, label = next(pair_generator)
                nm_of_one_batch.append(nm_1)
                nm_of_one_batch.append(nm_2)
                am_of_one_batch.append(am_1)
                am_of_one_batch.append(am_2)
                label_of_one_batch.append(label)
                sample_hash_of_one_batch.append(sample_hash_1)
                sample_hash_of_one_batch.append(sample_hash_2)
            # print(sample_hash_of_one_batch)
            batched_label = np.array(label_of_one_batch, dtype=int)
            yield self._pack_batch(nm_of_one_batch, am_of_one_batch), batched_label

    def triplets(self, batch_size):
        triplet_generator = self._triplet_generator()
        num_batch_in_total = self.num_epoch * self.step_per_epoch
        for batch_idx in range(num_batch_in_total):
            nm_of_one_batch = []
            am_of_one_batch = []
            for sample_idx in range(batch_size):
                nm_1, am_1, nm_2, am_2, nm_3, am_3 = next(triplet_generator)
                nm_of_one_batch.append(nm_1)
                nm_of_one_batch.append(nm_2)
                nm_of_one_batch.append(nm_1)
                nm_of_one_batch.append(nm_3)
                am_of_one_batch.append(am_1)
                am_of_one_batch.append(am_2)
                am_of_one_batch.append(am_1)
                am_of_one_batch.append(am_3)
            yield self._pack_batch(nm_of_one_batch, am_of_one_batch)

    def single(self, batch_size, if_shuffle=True):
        shuffled_sample_hash_collection = list(self.sample_hash_to_label_dict.keys())
        for epoch_idx in range(self.num_epoch):
            if if_shuffle:
                random.shuffle(shuffled_sample_hash_collection)
            nm_of_one_batch = []
            am_of_one_batch = []
            sample_hash_of_one_batch = []
            label_of_one_batch = []
            for sample_idx, sample_hash in enumerate(shuffled_sample_hash_collection):
                sample_label = self.sample_hash_to_label_dict[sample_hash]
                # print('sample_hash = {}; sample_label = {}'.format(sample_hash, sample_label))
                nm_path = os.path.join(self.nm_dir, sample_hash + '.nm')
                am_path = os.path.join(self.am_dir, sample_hash + '.am')
                try:
                    nm = np.loadtxt(nm_path, delimiter=',')
                except ValueError:
                    continue
                am = np.loadtxt(am_path, delimiter=',').astype(int)
                if len(am.shape) == 1:
                    am = am[None, :]
                if nm.shape[0] >= self.max_num_node_of_one_graph:
                    continue
                nm_of_one_batch.append(nm)
                am_of_one_batch.append(am)
                label_of_one_batch.append(sample_label)
                sample_hash_of_one_batch.append(sample_hash)
                if len(nm_of_one_batch) == batch_size:
                    batched_label = np.array(label_of_one_batch)
                    yield self._pack_batch(nm_of_one_batch, am_of_one_batch), batched_label
                    nm_of_one_batch = []
                    am_of_one_batch = []
                    sample_hash_of_one_batch = []
                    label_of_one_batch = []

    @property
    def num_validate_sample(self):
        num_validate_sample = 0
        for sample_hash in self.sample_hash_to_label_dict.keys():
            nm_path = os.path.join(self.nm_dir, sample_hash + '.nm')
            try:
                nm = np.loadtxt(nm_path, delimiter=',')
            except ValueError:
                continue
            if nm.shape[0] < self.max_num_node_of_one_graph:
                num_validate_sample += 1
        return num_validate_sample

    def _pack_batch(self, nm_list, ad_list):
        '''
        :param graphs: a list of (nm_matrix[num_nodes, node_feature_dims], am_matrix[num_edge, 2]) pairs. nm/am_matrixes are all numpy array.
        :return:
        '''
        num_node_list = [0]
        num_edge_list = []
        total_num_node = 0
        total_num_edge = 0
        batch_size = len(nm_list)
        for nm, am in zip(nm_list, ad_list):
            num_node_of_this_graph = nm.shape[0]
            num_node_list.append(num_node_of_this_graph)
            total_num_node += num_node_of_this_graph
            num_edge_of_this_graph = am.shape[0]
            num_edge_list.append(num_edge_of_this_graph)
            total_num_edge += num_edge_of_this_graph
        node_features = np.concatenate(nm_list, axis=0)
        cumsum = np.cumsum(num_node_list)
        indices = np.repeat(np.arange(batch_size), num_edge_list)  # [num_edge_this_batch]
        scattered = cumsum[indices]  # [num_edge_this_batch, ]


        edges = np.concatenate(ad_list, axis=0)
        edges[..., 0] += scattered
        edges[..., 1] += scattered

        edge_features = np.zeros(shape=(total_num_edge, self.inital_edge_feature_dim), dtype=np.float32)
        edge_features[np.arange(total_num_edge), edges[:, 2]] = 1

        return GraphData(from_idx=edges[..., 0],
                         to_idx=edges[..., 1],
                         node_features=node_features,
                         edge_features=edge_features,
                         graph_idx=np.repeat(np.arange(batch_size), np.array(num_node_list[1:])),
                         n_graphs=batch_size
                         )
