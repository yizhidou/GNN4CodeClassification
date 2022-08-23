from typing import List
import os
import numpy as np
from utils import *
from loss import euclidean_distance
import random, json, argparse
from graph_embedding_calculation import GraphEmbeddingCalculaor


class TestTaskBasic: # base class for multiple testing tasks
    def __init__(self, model_params_savepath, device):
        self.device = device
        self.model_params_savepath = model_params_savepath
        with open(model_params_savepath) as params_loader:
            self.model_params = json.load(params_loader)
        self.model_id = model_params_savepath.rsplit('/')[-1][:-5]
        # self.model_indicator = self.ckpt_path.rsplit('/', 1)[-1][:-5]
        self.nm_dir = self.model_params['dataset_params']['nm_dir']
        self.am_dir = self.model_params['dataset_params']['am_dir']
        self.dataset_name = self.model_params['dataset_params']['dataset_name']
        self.label_loader = LabelLoader(dataset_name=self.dataset_name,
                                        cwe_to_label_mapping_path=self.model_params['dataset_params'][
                                            'cwe_to_label_mapping_record'])
        with open(self.model_params['dataset_params']['train_vali_test_divide_record']) as divide_loader:
            self.train_vali_test_divide_record = json.load(divide_loader)
        # self.sample_fn_list_under_test = self.train_vali_test_divide_record['test_sha']
        self.ckpt_path = os.path.join(self.model_params['ckpt_save_dir'], self.model_id + '.ckpt')

        graph_embedding_record_savedir = ''
        if not os.path.isdir(graph_embedding_record_savedir):
            os.system('mkdir -p {}'.format(graph_embedding_record_savedir))
        self.graph_embedding_record_savepath = os.path.join(graph_embedding_record_savedir,
                                                            self.model_id + '_embeddings.json')
        if self.model_params['graph_embedding_net_settings']['prop_type'] == 'embedding':
            print('prop type is embedding, so graph embeddings could be calculated first~')
            if os.path.isfile(self.graph_embedding_record_savepath):
                print('embeddings found! yeah~')
            else:
                print('{} does not exit~ so we calculate them for future use~'.format(
                    self.graph_embedding_record_savepath))
                GraphEmbeddingCalculaor(model_params_path=model_params_savepath,
                                        record_savepath=self.graph_embedding_record_savepath,
                                        device=self.device).dump_to_record()
                print('embedding calculation has done~')
            with open(self.graph_embedding_record_savepath) as json_loader:
                self.sample_fn_to_graph_embedding_dict = json.load(json_loader)
        else:
            assert self.model_params['graph_embedding_net_settings']['prop_type'] == 'matching'
            print('prop type is matching, so there is no graph embeddings reserved')
            self.sample_fn_to_graph_embedding_dict = None

        # self.model = self.get_loaded_model_from_ckpt()  # actually, you don't even need to load a model if all embeddings are avaliable

    def get_loaded_model_from_ckpt(self):
        model, _ = build_model(self.model_params)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def get_graph_embedding_for_one_sample(self, sample_sha):
        sample_nm_path = os.path.join(self.nm_dir, sample_sha + '.nm')
        sample_am_path = os.path.join(self.am_dir, sample_sha + '.am')
        sample_nm = np.loadtxt(sample_nm_path, delimiter=',').astype(np.float32)
        sample_am = np.loadtxt(sample_am_path, delimiter=',').astype(int)
        edge_feature = np.zeros(shape=(sample_am.shape[0], 13), dtype=np.float32)
        from_idx = sample_am[:, 0]
        to_idx = sample_am[:, 1]
        graph_idx = np.zeros(shape=(sample_nm.shape[0],), dtype=np.int64)

        # model.to(self.device)
        # model.eval()
        with torch.no_grad():
            sample_graph_embedding = self.model(torch.from_numpy(sample_nm).to(self.device),
                                                torch.from_numpy(edge_feature).to(self.device),
                                                torch.from_numpy(from_idx).to(self.device),
                                                torch.from_numpy(to_idx).to(self.device),
                                                torch.from_numpy(graph_idx).to(self.device), 1)
        return sample_graph_embedding

    def get_graph_embeddings_for_samples(self, sample_fn_list):
        nm_list = []
        am_list = []
        exist_sample_fn_list = []
        for sample_sha in sample_fn_list:
            sample_nm_path = os.path.join(self.nm_dir, sample_sha + '.nm')
            sample_am_path = os.path.join(self.am_dir, sample_sha + '.am')
            if not os.path.isfile(sample_nm_path):
                continue
            sample_nm = np.loadtxt(sample_nm_path, delimiter=',').astype(np.float32)
            sample_am = np.loadtxt(sample_am_path, delimiter=',').astype(int)
            nm_list.append(sample_nm)
            am_list.append(sample_am)
            exist_sample_fn_list.append(sample_sha)
        packed_data = pack_datapoint(nm_list=nm_list,
                                     am_list=am_list)
        node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(packed_data)
        with torch.no_grad():
            graph_embeddings = self.model(node_features.to(self.device), edge_features.to(self.device),
                                          from_idx.to(self.device),
                                          to_idx.to(self.device),
                                          graph_idx.to(self.device), len(nm_list))
        # print('length of sample_{} = {}'.format(sample_fn_list, euclidean_distance(graph_embeddings, 0)))
        return graph_embeddings, np.array(exist_sample_fn_list, dtype=object)

    def batched_graph_embedding_generator(self, batch_size):
        batched_sample_fn_list = []
        batch_idx = 0
        if self.sample_fn_to_graph_embedding_dict:
            batched_graph_embedding_list = []
            for sample_sha in self.sample_fn_list_under_test:
                batch_idx += 1
                if len(batched_sample_fn_list) % batch_size == 0 and len(batched_sample_fn_list) > 0:
                    graph_embeddings_of_one_batch = torch.from_numpy(np.stack(batched_graph_embedding_list)).to(
                        self.device)
                    yield graph_embeddings_of_one_batch, np.array(batched_sample_fn_list, dtype=object)
                    batched_sample_fn_list = []
                    batched_graph_embedding_list = []
                else:
                    if not sample_sha in self.sample_fn_to_graph_embedding_dict:
                        continue
                    batched_sample_fn_list.append(sample_sha)
                    batched_graph_embedding_list.append(np.array(self.sample_fn_to_graph_embedding_dict[sample_sha]))
            if len(batched_sample_fn_list) > 0:
                graph_embeddings_of_last_batch = torch.from_numpy(np.stack(batched_graph_embedding_list)).to(
                    self.device)
                yield graph_embeddings_of_last_batch, np.array(batched_sample_fn_list, dtype=object)
        else:
            for sample_sha in self.sample_fn_list_under_test:
                batch_idx += 1
                if len(batched_sample_fn_list) % batch_size == 0 and len(batched_sample_fn_list) > 0:
                    # print('{}: embedding of {} is generating...'.format(sample_fn_idx, batched_sample_fn_list))
                    yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)
                    batched_sample_fn_list = []
                else:
                    batched_sample_fn_list.append(sample_sha)
            # print('the num of samples of type_{} is: {}'.format(debug_label, sample_fn_idx))
            if len(batched_sample_fn_list) > 0:
                yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)

    def batched_graph_generator(self, sample_fn_list, batch_size):
        batch_sample_fn_list = []
        batch_nm_list = []
        batch_am_list = []
        for sample_sha in sample_fn_list:
            sample_nm_path = os.path.join(self.nm_dir, sample_sha + '.nm')
            sample_am_path = os.path.join(self.am_dir, sample_sha + '.am')
            if not os.path.isfile(sample_nm_path):
                continue
            try:
                sample_nm = np.loadtxt(sample_nm_path, delimiter=',').astype(np.float32)
            except ValueError:
                continue
            sample_am = np.loadtxt(sample_am_path, delimiter=',').astype(int)
            batch_nm_list.append(sample_nm)
            batch_am_list.append(sample_am)
            batch_sample_fn_list.append(sample_sha)
            if len(batch_sample_fn_list) == batch_size:
                # packed_data = pack_datapoint(nm_list=batch_nm_list,
                #                              am_list=batch_am_list)
                yield batch_nm_list, batch_am_list, batch_sample_fn_list
                batch_sample_fn_list = []
                batch_nm_list = []
                batch_am_list = []

