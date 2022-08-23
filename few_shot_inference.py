from graph_embedding_calculation import GraphEmbeddingCalculatorForFewLables, \
    GraphEmbeddingCalculatorForPojWithJulietModel
from utils import *
import os, random, argparse
import json
from loss import *


class FewShotInference:
    def __init__(self, model_params_savepath, device, shot_num, division_version):
        self.device = device

        self.model_params_savepath = model_params_savepath
        with open(model_params_savepath) as params_loader:
            self.model_params = json.load(params_loader)
        self.model_id = model_params_savepath.rsplit('/')[-1][:-5]

        label_distribution_record_path = './auxiliary_records/juliet_cwe_label_distribution.json'
        with open(label_distribution_record_path) as json_loader:
            cwe_label_distribution = json.load(json_loader)
        self.few_cwe_label_collection = dict()
        total_sample_num = 0
        few_sample_num = 0
        for cwe_str, sample_num in cwe_label_distribution.items():
            total_sample_num += sample_num
            if sample_num < 1000:
                few_sample_num += sample_num
                self.few_cwe_label_collection[cwe_str] = sample_num
        print('There are in total {} samples; {} are of few samples and {} are of many samples'.format(total_sample_num,
                                                                                                       few_sample_num,
                                                                                                       total_sample_num - few_sample_num))
        print('There are {} out of {} types of samples that are few'.format(len(self.few_cwe_label_collection),
                                                                            len(cwe_label_distribution)))

        few_embedding_record_dir = ''
        few_embedding_filename = '{}_few_embeddings.json'.format(self.model_id)
        self.few_embedding_record_path = os.path.join(few_embedding_record_dir, few_embedding_filename)
        if os.path.isfile(self.few_embedding_record_path):
            with open(self.few_embedding_record_path) as embedding_loader:
                self.few_sample_fn_to_embedding = json.load(embedding_loader)
            print('few embedding record found! yeah~')
        else:
            print('few embedding record not found, so we calculate them first for future use~')
            few_sample_fn_list = []
            for nm_filename in os.listdir(self.model_params['dataset_params']['nm_dir']):
                sample_fn_str = nm_filename[:-3]
                sample_cwe_str = 'CWE-' + sample_fn_str.split('.')[-2]
                if sample_cwe_str in self.few_cwe_label_collection:
                    few_sample_fn_list.append(sample_fn_str)
            print('In total, there are {} samples of few labels'.format(len(few_sample_fn_list)))
            graph_embedding_calculator_for_few_labels = GraphEmbeddingCalculatorForFewLables(
                model_params_savepath=model_params_savepath,
                few_embedding_record_savepath=self.few_embedding_record_path,
                sample_fn_list=few_sample_fn_list,
                device=self.device)
            self.few_sample_fn_to_embedding = graph_embedding_calculator_for_few_labels.dump_to_record()

        graph_embedding_record_savedir = ''
        self.graph_embedding_record_savepath = os.path.join(graph_embedding_record_savedir,
                                                            self.model_id + '_embeddings.json')
        assert os.path.isfile(self.graph_embedding_record_savepath)
        with open(self.graph_embedding_record_savepath) as json_loader:
            self.sample_fn_to_embedding = json.load(json_loader)

        few_inference_result_savedir = ''

        self.prototype_mode_few_inference_result_savepath = os.path.join(few_inference_result_savedir,
                                                                         '{}_shot_results'.format(shot_num),
                                                                         '{}_version_{}_{}_shot_inference_result.prototype_mode.json'.format(
                                                                             self.model_id, division_version, shot_num))

        self.division_version = division_version
        sq_division_result_savedir = 'juliet_support_query_set_division_records/'
        if not os.path.isdir(sq_division_result_savedir):
            os.system('mkdir -p {}'.format(sq_division_result_savedir))
        self.sq_division_result_savepath = os.path.join(sq_division_result_savedir,
                                                        '{}_shot_support_query_division_v{}.json'.format(shot_num,
                                                                                                         division_version))
        if not os.path.isfile(self.sq_division_result_savepath):
            self.support_query_division(num_support=shot_num)

        # def calculate_cluster_mean_from_fn_to_embedding_dict(self, sha_to_embedding_dict):
        cwe_str_to_embeddings_dict = dict()
        for sample_fn_str, embedding_in_np_array in self.sample_fn_to_embedding.items():
            sample_cwe_str = 'CWE-' + sample_fn_str.split('.')[-2]
            if not sample_cwe_str in cwe_str_to_embeddings_dict:
                cwe_str_to_embeddings_dict[sample_cwe_str] = []
            cwe_str_to_embeddings_dict[sample_cwe_str].append(embedding_in_np_array)
        self.prototype_corresponding_cwe_str_list = list()
        self.prototype_embedding_list = list()
        for cwe_str, embedding_list_of_this_cwe in cwe_str_to_embeddings_dict.items():
            self.prototype_corresponding_cwe_str_list.append(cwe_str)
            self.prototype_embedding_list.append(np.mean(np.stack(arrays=embedding_list_of_this_cwe, axis=0), axis=0))

    def few_shot_inference_for_one_label_prototype_mode(self, cwe_str, support_sample_fn_list, query_sample_fn_list):
        print('{} is on processing'.format(cwe_str))

        prototype_embedding_list = self.prototype_embedding_list.copy()
        prototype_corresponding_cwe_str_list = self.prototype_corresponding_cwe_str_list.copy()

        support_embedding_list = list()
        for support_sample_fn in support_sample_fn_list:
            if not support_sample_fn in self.few_sample_fn_to_embedding:
                continue
            support_embedding_list.append(self.few_sample_fn_to_embedding[support_sample_fn])

        prototype_from_support_set = np.mean(np.stack(support_embedding_list, axis=0), axis=0)
        prototype_embedding_list.append(prototype_from_support_set)
        prototype_corresponding_cwe_str_list.append(cwe_str)
        prototype_sample_embeddings = torch.from_numpy(np.stack(prototype_embedding_list)).to(self.device)
        prototype_cwe_label = np.array(prototype_corresponding_cwe_str_list, dtype=object)
        del prototype_embedding_list
        del prototype_corresponding_cwe_str_list

        result = dict()
        for query_sample_fn in query_sample_fn_list:
            print('{} is on quering...'.format(query_sample_fn))
            if not query_sample_fn in self.few_sample_fn_to_embedding:
                print('this sample skip~')
                continue
            test_embedding = torch.from_numpy(np.array(self.few_sample_fn_to_embedding[query_sample_fn])).to(
                self.device)
            # [num_features, ]
            similarity_vector = -euclidean_distance(prototype_sample_embeddings, test_embedding)
            # [num_train_set, num_features], [num_features, ] -> [num_features, ]
            sorted_indices = torch.argsort(similarity_vector, dim=-1, descending=True).cpu().detach().numpy()
            # [num_features, ]
            sorted_similarity = similarity_vector[sorted_indices]
            # result[query_sample_fn] = [(prototype_cwe_label[sorted_indices[idx].item()],
            #                              sorted_similarity[sorted_indices[idx].item()].cpu().detach().numpy().item())
            #                             for
            #                             idx in
            #                             range(10)]
            result[query_sample_fn] = [(prototype_cwe_label[sorted_idx],
                                         sorted_similarity[sorted_idx].cpu().detach().numpy().item())
                                        for
                                        sorted_idx in
                                        sorted_indices]
        return result

    def few_shot_inference_for_every_label_prototype_mode(self):
        print('prototype mode is on')
        assert os.path.isfile(self.sq_division_result_savepath), 'division record does not exist! check plz!'
        with open(self.sq_division_result_savepath) as sq_loader:
            sq_division_dict = json.load(sq_loader)

        result = dict()
        for few_label_cwe_str, sq_division_dict_of_this_label in sq_division_dict.items():
            result[few_label_cwe_str] = self.few_shot_inference_for_one_label_prototype_mode(few_label_cwe_str,
                                                                                             sq_division_dict_of_this_label[
                                                                                                 'support'],
                                                                                             sq_division_dict_of_this_label[
                                                                                                 'query'])

            with open(self.prototype_mode_few_inference_result_savepath, 'w') as result_dumper:
                json.dump(result, result_dumper, indent=3)

    def get_result_summary_prototype_mode(self):
        assert os.path.isfile(self.prototype_mode_few_inference_result_savepath), 'no result exist!'
        with open(self.prototype_mode_few_inference_result_savepath) as result_loader:
            inference_result = json.load(result_loader)
        total_sample_count = 0
        hit_1_in_total = 0
        hit_2_in_total = 0
        hit_3_in_total = 0
        mean_rank_in_toal = 0
        for cwe_label_str, inference_result_of_this_cwe in inference_result.items():
            print('{}~~~~~~~~~~~~'.format(cwe_label_str))
            sample_count_of_this_cwe = 0
            hit_with_top = 0
            hit_within_2 = 0
            hit_within_3 = 0
            rank_sum_of_this_cwe = 0
            for sample_fn_str, matches in inference_result_of_this_cwe.items():
                sample_count_of_this_cwe += 1
                real_label = 'CWE-' + sample_fn_str.split('.')[-2]
                top_1_near_cwe_str = matches[0][0]
                if real_label == top_1_near_cwe_str:
                    # print('sample {} matches predicted label {}'.format(sample_fn, matches[0][0]))
                    hit_with_top += 1
                    hit_within_2 += 1
                    hit_within_3 += 1
                    rank_sum_of_this_cwe += 1
                else:
                    for match_idx, match in enumerate(matches):
                        if match[0] == real_label:
                            rank_sum_of_this_cwe += match_idx + 1
                            if match_idx < 3:
                                hit_within_3 += 1
                                if match_idx < 2:
                                    hit_within_2 += 1
                            break
            mean_rank_in_toal += rank_sum_of_this_cwe
            total_sample_count += sample_count_of_this_cwe
            hit_1_in_total += hit_with_top
            hit_2_in_total += hit_within_2
            hit_3_in_total += hit_within_3
            print(
                'in total there are {} samples on detection of label {}, {} are predicted correctly if only look into the nearest prediction,\n{} are predicted correctly if only look into the top_2 prediction,\n{} are predicted correctly if only look into top_3 prediction,\navg h@1 = {}; avg h@2 = {}; avg h@3 = {}\nMR= {}'.format(
                    sample_count_of_this_cwe,
                    cwe_label_str,
                    hit_with_top,
                    hit_within_2,
                    hit_within_3,
                    hit_with_top / sample_count_of_this_cwe,
                    hit_within_2 / sample_count_of_this_cwe,
                    hit_within_3 / sample_count_of_this_cwe,
                    rank_sum_of_this_cwe / sample_count_of_this_cwe))
        print('over all cwe labels: H@1 = {} H@2 = {} H@3 = {} MR = {}'.format(hit_1_in_total / total_sample_count,
                                                                               hit_2_in_total / total_sample_count,
                                                                               hit_3_in_total / total_sample_count,
                                                                               mean_rank_in_toal / total_sample_count))

    def support_query_division(self, num_support):
        random.seed(seed + self.division_version)
        label_distribution_record_path = './auxiliary_records/juliet_cwe_label_distribution.json'
        with open(label_distribution_record_path) as json_loader:
            cwe_label_distribution = json.load(json_loader)
        few_cwe_label_collection = dict()
        for cwe_str, sample_num in cwe_label_distribution.items():
            if sample_num < 1000:
                few_cwe_label_collection[cwe_str] = sample_num
        print('There are {} types of samples that are few'.format(len(few_cwe_label_collection)))

        cwe_label_to_sample_fn_collection = dict()
        for nm_filename in os.listdir(self.model_params['dataset_params']['nm_dir']):
            if not nm_filename.endswith('.nm'):
                continue
            sample_fn_str = nm_filename[:-3]
            sample_cwe_str = 'CWE-' + sample_fn_str.split('.')[-2]
            if sample_cwe_str in few_cwe_label_collection:
                if sample_cwe_str not in cwe_label_to_sample_fn_collection:
                    cwe_label_to_sample_fn_collection[sample_cwe_str] = []
                cwe_label_to_sample_fn_collection[sample_cwe_str].append(sample_fn_str)

        sq_division_result = dict()
        for cwe_label in few_cwe_label_collection.keys():
            sq_division_result[cwe_label] = dict()
            sq_division_result[cwe_label]['support'] = random.choices(
                population=cwe_label_to_sample_fn_collection[cwe_label],
                k=num_support)
            sq_division_result[cwe_label]['query'] = [sample_fn for sample_fn in
                                                      cwe_label_to_sample_fn_collection[cwe_label] if
                                                      not sample_fn in sq_division_result[cwe_label]['support']]
        with open(self.sq_division_result_savepath, 'w') as result_dumper:
            json.dump(sq_division_result, result_dumper, indent=3)


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
    parser.add_argument('--division_version', type=int)
    parser.add_argument('--shot_num', type=int)
    parser.add_argument('--model_params_path', type=str)
    args = parser.parse_args()
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    f = FewShotInference(model_params_savepath=args.model_params_path, device=device,
                         shot_num=args.shot_num,
                         division_version=args.division_version)
    # if not os.path.isfile(f.prototype_mode_few_inference_result_savepath):
    f.few_shot_inference_for_every_label_prototype_mode()
    # f.get_result_summary_prototype_mode()
    # else:
    #     f.get_result_summary_prototype_mode()
