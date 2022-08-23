from model_test import TestTaskBasic
import os, random, argparse
from utils import *
from loss import euclidean_distance
from dataset_related_utils import *


class KNNCheckTask(TestTaskBasic):
    def __init__(self, model_params_savepath, device):
        TestTaskBasic.__init__(self, model_params_savepath=model_params_savepath, device=device)
        test_result_savedir = ''
        if not os.path.isdir(test_result_savedir):
            os.system('mkdir -p {}'.format(test_result_savedir))
        self.test_record_filepath = os.path.join(test_result_savedir, self.model_id + '_knn_matchresult.json')

    def calculate_knn(self):
        result = {}
        if self.sample_fn_to_graph_embedding_dict != None:
            assert self.model_params['graph_embedding_net_settings']['prop_type'] == 'embedding'
            train_samples_embeddings_list = []
            exist_train_samples_fn_list = []
            for train_sample_sha in self.train_vali_test_divide_record['train_sha']:
                if train_sample_sha in self.sample_fn_to_graph_embedding_dict:
                    exist_train_samples_fn_list.append(train_sample_sha)
                    train_samples_embeddings_list.append(
                        np.array(self.sample_fn_to_graph_embedding_dict[train_sample_sha]))
            train_samples_embeddings = torch.from_numpy(np.stack(train_samples_embeddings_list)).to(self.device)
            train_samples_shas = np.array(exist_train_samples_fn_list, dtype=object)
            del train_samples_embeddings_list
            del exist_train_samples_fn_list

            for test_sha in self.train_vali_test_divide_record['test_sha']:
                if not test_sha in self.sample_fn_to_graph_embedding_dict:
                    continue
                print('{} is on testing...'.format(test_sha))
                test_embedding = torch.from_numpy(np.array(self.sample_fn_to_graph_embedding_dict[test_sha])).to(
                    self.device)
                # [num_features, ]
                similarity_vector = -euclidean_distance(train_samples_embeddings, test_embedding)
                # [num_train_set, num_features], [num_features, ] -> [num_features, ]
                sorted_indices = torch.argsort(similarity_vector, dim=-1, descending=True).cpu().detach().numpy()
                # [num_features, ]
                sorted_similarity = similarity_vector[sorted_indices]
                result[test_sha] = [(train_samples_shas[sorted_indices[idx].item()],
                                     sorted_similarity[sorted_indices[idx].item()].cpu().detach().numpy().item()) for
                                    idx in
                                    range(10)]
            with open(self.test_record_filepath, 'w') as json_logger:
                json.dump(result, json_logger, indent=3)
        else:
            assert self.model_params['graph_embedding_net_settings']['prop_type'] == 'matching'

            kcentroids_prototypes_record_savepath = ''
            assert os.path.isfile(kcentroids_prototypes_record_savepath)
            print('prototypes found~ so we match on these prototypes!!!')
            prototype_fn_list = []
            with open(kcentroids_prototypes_record_savepath) as prototype_loader:
                prototype_dict = json.load(prototype_loader)
                for label, corresponding_prototypes in prototype_dict.items():
                    prototype_fn_list += corresponding_prototypes
            if os.path.isfile(self.test_record_filepath):
                print('part of the knn match result exist! so we resume~')
                with open(self.test_record_filepath) as result_loader:
                    result = json.load(result_loader)

            model = self.get_loaded_model_from_ckpt()
            # batch_size = 1
            total_num_of_test_samples = len(self.train_vali_test_divide_record['test_sha'])
            print('in total there are {} test samples'.format(total_num_of_test_samples))
            # exit(666)
            for test_idx, test_sha in enumerate(self.train_vali_test_divide_record['test_sha']):
                print('{}/{}: {} is on testing...'.format(test_idx, total_num_of_test_samples, test_sha))
                if test_sha in result:
                    print('matched~ so skip~~')
                    continue
                test_nm_path = os.path.join(self.nm_dir, test_sha + '.nm')
                test_am_path = os.path.join(self.am_dir, test_sha + '.am')
                if not os.path.isfile(test_nm_path):
                    continue
                try:
                    test_nm = np.loadtxt(test_nm_path, delimiter=',').astype(np.float32)
                except ValueError:
                    continue
                test_am = np.loadtxt(test_am_path, delimiter=',').astype(int)
                concat_nm_list = []
                concat_am_list = []
                sample_fn_list = []
                # most_similar_embedding_list = []
                most_similar_fn_list = []
                candi_similarity_list = []
                for prototype_sha in prototype_fn_list:
                    prototype_nm_path = os.path.join(self.nm_dir, prototype_sha + '.nm')
                    prototype_am_path = os.path.join(self.am_dir, prototype_sha + '.am')
                    if not os.path.isfile(prototype_nm_path):
                        continue
                    try:
                        sample_nm = np.loadtxt(prototype_nm_path, delimiter=',').astype(np.float32)
                    except ValueError:
                        continue
                    sample_am = np.loadtxt(prototype_am_path, delimiter=',').astype(int)
                    concat_nm_list.append(sample_nm)
                    concat_nm_list.append(test_nm)
                    concat_am_list.append(sample_am)
                    concat_am_list.append(test_am)
                    sample_fn_list.append(prototype_sha)
                    # if len(batch_sample_fn_list) == batch_size:

                    packed_data = pack_datapoint(nm_list=concat_nm_list,
                                                 am_list=concat_am_list)
                    concat_node_features, concat_edge_features, concat_from_idx, concat_to_idx, concat_graph_idx = get_graph(
                        packed_data)
                    try:
                        _, concat_graph_vectors = model(concat_node_features.to(self.device),
                                                        concat_edge_features.to(self.device),
                                                        concat_from_idx.to(self.device),
                                                        concat_to_idx.to(self.device),
                                                        concat_graph_idx.to(self.device), 2)
                    except:
                        print('some happend~ so skip')
                        break
                    train_samples_embeddings = torch.index_select(input=concat_graph_vectors,
                                                                  dim=0,
                                                                  index=torch.arange(0, 2, 2).to(
                                                                      self.device))
                    test_embeddings = torch.index_select(input=concat_graph_vectors,
                                                         dim=0,
                                                         index=torch.arange(1, 2, 2).to(self.device))

                    batched_similarity = -euclidean_distance(train_samples_embeddings, test_embeddings)
                    # [batch_size, num_features], [batch_size, num_features] -> [batch_size]
                    sorted_indices = torch.argsort(batched_similarity, dim=-1,
                                                   descending=True).cpu().detach().numpy()
                    # [batch_size, ]
                    # print('the embedding of train_sample = {}'.format(
                    #     concat_graph_vectors.cpu().detach().numpy()[2 * sorted_indices[0]]))
                    # print('the embedding of test_sample = {}'.format(
                    #     concat_graph_vectors.cpu().detach().numpy()[2 * sorted_indices[0] + 1]))
                    # print('sorted_indices.shape = {}'.format(sorted_indices.shape))
                    # print('sorted_indices = {}'.format(sorted_indices))
                    batched_similarity = batched_similarity.cpu().detach().numpy()
                    # print('batched_similarity = {}\n type={}'.format(batched_similarity, type(batched_similarity)))
                    candi_similarity_list.append(batched_similarity[sorted_indices[0]])
                    # print('candi sim = {}'.format(batched_similarity[sorted_indices[0]]))
                    # print('the candi sim in this batch is: {}'.format(batched_similarity[sorted_indices[0]]))
                    sorted_similarity = batched_similarity[sorted_indices]
                    # print('sorted_similarity = {}'.format(sorted_similarity))
                    most_similar_fn_list.append(sample_fn_list[sorted_indices[0]])
                    # print('batch_sample_fn_list = {}'.format(batch_sample_fn_list))
                    # print('the candi sha in this batch is: {}'.format(batch_sample_fn_list[sorted_indices[0]]))
                    concat_nm_list = []
                    concat_am_list = []
                    sample_fn_list = []
                    # if exit_flag == 10:
                    #     exit(666)
                    # else:
                    #     exit_flag += 1


                # most_similar_embeddings = torch.cat(tensors=most_similar_embedding_list, dim=0)
                most_similar_sha = np.array(most_similar_fn_list, dtype=object)
                candi_similarity = np.array(candi_similarity_list)
                sorted_indices = np.argsort(candi_similarity, axis=-1)

                result[test_sha] = [(most_similar_sha[sorted_indices[-1 - idx].item()],
                                     candi_similarity[sorted_indices[-1 - idx].item()].item()) for
                                    idx in
                                    range(10)]

                with open(self.test_record_filepath, 'w') as json_logger:
                    json.dump(result, json_logger, indent=3)

    def get_knn_result_summary(self):
        assert os.path.isfile(self.test_record_filepath), 'no result exist!'
        label_loader = LabelLoader(dataset_name=self.dataset_name,
                                   cwe_to_label_mapping_path=self.model_params['dataset_params'][
                                       'cwe_to_label_mapping_record'])
        with open(self.test_record_filepath) as result_loader:
            knn_result = json.load(result_loader)
        sample_count = 0
        P_1 = 0
        P_5 = 0
        P_10 = 0
        for sample_sha, matches in knn_result.items():
            sample_count += 1
            real_label = label_loader.get_label(sample_sha)
            top_1_near_label = label_loader.get_label(matches[0][0])
            if real_label == top_1_near_label:
                # print('sample {} matches predicted label {}'.format(sample_sha, matches[0][0]))
                P_1 += 1
            correct_pred_among_top_5 = sum(
                [1 if real_label == label_loader.get_label(matches[idx][0]) else 0 for idx in range(5)])
            correct_pred_among_top_10 = correct_pred_among_top_5 + sum(
                [1 if real_label == label_loader.get_label(matches[idx][0]) else 0 for idx in range(5, 10)])
            P_5 += float(correct_pred_among_top_5) / 5.0
            P_10 += float(correct_pred_among_top_10) / 10.0
        print(
            'in total there are {} samples on detection, {} are predicted correctly if only look into the nearest prediction,\navg P@1 = {}; avg P@5 = {}; avg P@10 = {}'.format(
                sample_count,
                P_1,
                P_1 / sample_count,
                P_5 / sample_count,
                P_10 / sample_count))



def get_loaded_model_from_ckpt(model_params_path, model_params, model_id, device):
    model, _ = build_model(model_params)
    ckpt_path = os.path.join(model_params['ckpt_save_dir'], model_id + '.ckpt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        with open(model_params_path) as params_loader:
            model_params = json.load(params_loader)
        model, _ = build_model_deprecated(model_params)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    # Set random seeds
    seed = 6
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_params_path', type=str)
    args = parser.parse_args()
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    knn_check_task = KNNCheckTask(model_params_savepath=args.model_params_path, device=device)
    knn_check_task.calculate_knn()



