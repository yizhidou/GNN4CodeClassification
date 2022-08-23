from dataset_related_utils import *
from utils import *
import os, json, random, argparse


class GraphEmbeddingCalculaor:

    def __init__(self, model_params_path, record_savepath, device):
        self.model_params_path = model_params_path
        with open(model_params_path) as params_loader:
            self.model_params = json.load(params_loader)
        self.model_id = model_params_path.rsplit('/', 1)[-1][:-5]

        self.label_loader = LabelLoader(dataset_name=self.model_params['dataset_params']['dataset_name'],
                                        cwe_to_label_mapping_path=self.model_params['dataset_params'][
                                            'cwe_to_label_mapping_record'])
        self.nm_dir = self.model_params['dataset_params']['nm_dir']
        self.am_dir = self.model_params['dataset_params']['am_dir']

        self.ckpt_path = os.path.join(self.model_params['ckpt_save_dir'], self.model_id + '.ckpt')
        with open(self.model_params['dataset_params']['train_vali_test_divide_record']) as divide_loader:
            self.train_vali_test_divide_record = json.load(divide_loader)
        self.sample_fn_list = self.train_vali_test_divide_record['train_fn'] + self.train_vali_test_divide_record[
            'test_fn'] + self.train_vali_test_divide_record['validation_fn']
        self.record_savepath = record_savepath
        self.device = device
        self.model = self.get_loaded_model_from_ckpt()

    def get_loaded_model_from_ckpt(self):

        model, _ = build_model(self.model_params)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            with open(self.model_params_path) as params_loader:
                self.model_params = json.load(params_loader)
            model, _ = build_model_deprecated(self.model_params)
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def get_graph_embeddings_for_samples(self, sample_fn_list):
        nm_list = []
        am_list = []
        exist_sample_fn_list = []
        for sample_fn in sample_fn_list:
            sample_nm_path = os.path.join(self.nm_dir, sample_fn + '.nm')
            sample_am_path = os.path.join(self.am_dir, sample_fn + '.am')
            if not os.path.isfile(sample_nm_path):
                continue
            try:
                sample_nm = np.loadtxt(sample_nm_path, delimiter=',').astype(np.float32)
            except ValueError:
                continue
            sample_am = np.loadtxt(sample_am_path, delimiter=',').astype(int)
            nm_list.append(sample_nm)
            am_list.append(sample_am)
            exist_sample_fn_list.append(sample_fn)
        packed_data = pack_datapoint(nm_list=nm_list,
                                     am_list=am_list)
        node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(packed_data)
        with torch.no_grad():
            graph_embeddings_1, graph_embeddings_2 = self.model(node_features.to(self.device),
                                                                edge_features.to(self.device),
                                                                from_idx.to(self.device),
                                                                to_idx.to(self.device),
                                                                graph_idx.to(self.device), len(nm_list))
        if self.model_params['training_settings']['pair_or_triplet_or_single'] == 'single' and \
                self.model_params['graph_embedding_net_settings']['prop_type'] == 'embedding':
            graph_embeddings = graph_embeddings_1
        else:
            graph_embeddings = graph_embeddings_2
        assert graph_embeddings.shape[-1] == 128    #, 'shape of graph_embeddings: {}'.format(graph_embeddings.shape)
        return graph_embeddings, exist_sample_fn_list

    def batched_graph_embedding_generator(self, batch_size=256):
        batched_sample_fn_list = []
        batch_idx = 0
        for sample_fn in self.sample_fn_list:
            if len(batched_sample_fn_list) % batch_size == 0 and len(batched_sample_fn_list) > 0:
                print('batch_{}: embedding is generating...'.format(batch_idx))
                batch_idx += 1
                yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)
                batched_sample_fn_list = []
            else:
                batched_sample_fn_list.append(sample_fn)
        if len(batched_sample_fn_list) > 0:
            print('the last batch: embedding is generating...')
            yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)

    def dump_to_record(self):
        sample_fn_to_embedding_record = {}
        for batched_graph_embeddings, batched_sample_fns in self.batched_graph_embedding_generator():
            for sample_idx, sample_fn in enumerate(batched_sample_fns):
                sample_fn_to_embedding_record[sample_fn] = batched_graph_embeddings[sample_idx].cpu().numpy().tolist()
        with open(self.record_savepath, 'w') as recorder:
            json.dump(sample_fn_to_embedding_record, recorder, indent=3)


class GraphEmbeddingCalculatorForFewLables:
    def __init__(self, model_params_savepath, few_embedding_record_savepath, sample_fn_list, device):
        self.model_params_savepath = model_params_savepath
        with open(model_params_savepath) as params_loader:
            self.model_params = json.load(params_loader)
        self.model_id = model_params_savepath.rsplit('/')[-1][:-5]
        self.ckpt_path = os.path.join(self.model_params['ckpt_save_dir'], self.model_id + '.ckpt')
        self.nm_dir = self.model_params['dataset_params']['nm_dir']
        self.am_dir = self.model_params['dataset_params']['am_dir']
        self.sample_fn_list = sample_fn_list
        self.device = device
        self.model = self.get_loaded_model_from_ckpt()
        self.few_embedding_record_savepath = few_embedding_record_savepath

    def get_loaded_model_from_ckpt(self):
        model, _ = build_model(self.model_params)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            with open(self.model_params_savepath) as params_loader:
                self.model_params = json.load(params_loader)
            model, _ = build_model_deprecated(self.model_params)
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def batched_graph_embedding_generator(self, batch_size=256):
        batched_sample_fn_list = []
        batch_idx = 0
        for sample_fn in self.sample_fn_list:
            if len(batched_sample_fn_list) % batch_size == 0 and len(batched_sample_fn_list) > 0:
                print('batch_{}: embedding is generating...'.format(batch_idx))
                batch_idx += 1
                yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)
                batched_sample_fn_list = []
            else:
                batched_sample_fn_list.append(sample_fn)
        if len(batched_sample_fn_list) > 0:
            print('the last batch: embedding is generating...')
            yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)

    def get_graph_embeddings_for_samples(self, sample_fn_list):
        nm_list = []
        am_list = []
        exist_sample_fn_list = []
        for sample_fn in sample_fn_list:
            sample_nm_path = os.path.join(self.nm_dir, sample_fn + '.nm')
            sample_am_path = os.path.join(self.am_dir, sample_fn + '.am')
            if not os.path.isfile(sample_nm_path):
                continue
            try:
                sample_nm = np.loadtxt(sample_nm_path, delimiter=',').astype(np.float32)
            except ValueError:
                continue
            sample_am = np.loadtxt(sample_am_path, delimiter=',').astype(int)
            nm_list.append(sample_nm)
            am_list.append(sample_am)
            exist_sample_fn_list.append(sample_fn)
        packed_data = pack_datapoint(nm_list=nm_list,
                                     am_list=am_list)
        node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(packed_data)
        with torch.no_grad():
            graph_embeddings_1, graph_embeddings_2 = self.model(node_features.to(self.device),
                                                                edge_features.to(self.device),
                                                                from_idx.to(self.device),
                                                                to_idx.to(self.device),
                                                                graph_idx.to(self.device), len(nm_list))
        if self.model_params['training_settings']['pair_or_triplet_or_single'] == 'single' and \
                self.model_params['graph_embedding_net_settings']['prop_type'] == 'embedding':
            graph_embeddings = graph_embeddings_1
        else:
            graph_embeddings = graph_embeddings_2
        assert graph_embeddings.shape[-1] == 128
        return graph_embeddings, exist_sample_fn_list

    def dump_to_record(self):
        sample_fn_to_embedding_record = {}
        for batched_graph_embeddings, batched_sample_fns in self.batched_graph_embedding_generator():
            for sample_idx, sample_fn in enumerate(batched_sample_fns):
                sample_fn_to_embedding_record[sample_fn] = batched_graph_embeddings[sample_idx].cpu().numpy().tolist()
        with open(self.few_embedding_record_savepath, 'w') as recorder:
            json.dump(sample_fn_to_embedding_record, recorder, indent=3)
        return sample_fn_to_embedding_record


class GraphEmbeddingCalculatorForPojWithJulietModel:
    def __init__(self, model_params_path, poj_embedding_calculated_with_juliet_model_savedir, poj_sample_fn_list, device):
        self.model_params_path = model_params_path
        with open(model_params_path) as params_loader:
            self.model_params = json.load(params_loader)
        self.model_id = model_params_path.rsplit('/', 1)[-1][:-5]

        self.poj_dataset_dir = ''
        self.ckpt_path = os.path.join(self.model_params['ckpt_save_dir'], self.model_id + '.ckpt')

        self.poj_sample_fn_list = poj_sample_fn_list
        self.record_savepath = record_savepath
        self.device = device
        self.model = self.get_loaded_model_from_ckpt()

    def get_loaded_model_from_ckpt(self):

        model, _ = build_model(self.model_params)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            with open(self.model_params_path) as params_loader:
                self.model_params = json.load(params_loader)
            model, _ = build_model_deprecated(self.model_params)
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def get_graph_embeddings_for_samples(self, sample_fn_list):
        nm_list = []
        am_list = []
        exist_sample_fn_list = []
        for sample_fn in sample_fn_list:
            sample_nm_path = os.path.join(self.poj_dataset_dir, sample_fn + '.nm')
            sample_am_path = os.path.join(self.poj_dataset_dir, sample_fn + '.am')
            if not os.path.isfile(sample_nm_path):
                continue
            try:
                sample_nm = np.loadtxt(sample_nm_path, delimiter=',').astype(np.float32)
            except ValueError:
                continue
            sample_am = np.loadtxt(sample_am_path, delimiter=',').astype(int)
            nm_list.append(sample_nm)
            am_list.append(sample_am)
            exist_sample_fn_list.append(sample_fn)
        packed_data = pack_datapoint(nm_list=nm_list,
                                     am_list=am_list)
        node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(packed_data)
        with torch.no_grad():
            graph_embeddings_1, graph_embeddings_2 = self.model(node_features.to(self.device),
                                                                edge_features.to(self.device),
                                                                from_idx.to(self.device),
                                                                to_idx.to(self.device),
                                                                graph_idx.to(self.device), len(nm_list))
        if self.model_params['training_settings']['pair_or_triplet_or_single'] == 'single' and \
                self.model_params['graph_embedding_net_settings']['prop_type'] == 'embedding':
            graph_embeddings = graph_embeddings_1
        else:
            graph_embeddings = graph_embeddings_2
        assert graph_embeddings.shape[-1] == 128
        return graph_embeddings, exist_sample_fn_list

    def batched_graph_embedding_generator(self, batch_size=256):
        batched_sample_fn_list = []
        batch_idx = 0
        for sample_fn in self.poj_sample_fn_list:
            if len(batched_sample_fn_list) % batch_size == 0 and len(batched_sample_fn_list) > 0:
                print('batch_{}: embedding is generating...'.format(batch_idx))
                batch_idx += 1
                yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)
                batched_sample_fn_list = []
            else:
                batched_sample_fn_list.append(sample_fn)
        if len(batched_sample_fn_list) > 0:
            print('the last batch: embedding is generating...')
            yield self.get_graph_embeddings_for_samples(sample_fn_list=batched_sample_fn_list)

    def dump_to_record(self):
        sample_fn_to_embedding_record = {}
        for batched_graph_embeddings, batched_sample_fns in self.batched_graph_embedding_generator():
            for sample_idx, sample_fn in enumerate(batched_sample_fns):
                sample_fn_to_embedding_record[sample_fn] = batched_graph_embeddings[sample_idx].cpu().numpy().tolist()
        with open(self.record_savepath, 'w') as recorder:
            json.dump(sample_fn_to_embedding_record, recorder, indent=3)


