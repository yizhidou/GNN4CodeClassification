from dataset import YZDDataset
from graphembeddingnetwork import GraphEmbeddingNet, GraphEncoder, GraphAggregator, GraphAggregator_leaky
from graphmatchingnetwork import GraphMatchingNet
import copy
import torch
import numpy as np
from dataset import GraphData


def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def build_model(config):
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """

    encoder = GraphEncoder(node_feature_dim=config['encoder_settings']['node_feature_dim'],
                           edge_feature_dim=config['encoder_settings']['edge_feature_dim'],
                           node_hidden_sizes=config['encoder_settings']['node_hidden_sizes'],
                           edge_hidden_sizes=config['encoder_settings']['edge_hidden_sizes'])

    if config['graph_embedding_net_settings']['prop_type'] == 'embedding':
        # without cross-graph attention
        embedding_aggregator = GraphAggregator(node_state_dim=config['aggregator_settings']['node_state_dim'],
                                               node_hidden_sizes=config['aggregator_settings'][
                                                   'node_hidden_sizes'],
                                               graph_transform_sizes=config['aggregator_settings'][
                                                   'graph_transform_sizes'],
                                               gated=config['aggregator_settings']['gated'],
                                               aggregation_type=config['aggregator_settings']['gated'],
                                               use_transformer=config['aggregator_settings']['use_transformer'],
                                               use_mask=config['aggregator_settings']['use_mask'])
        model = GraphEmbeddingNet(encoder=encoder,
                                  aggregator=embedding_aggregator,
                                  node_state_dim=config['graph_embedding_net_settings']['node_state_dim'],
                                  edge_state_dim=config['graph_embedding_net_settings']['edge_state_dim'],
                                  edge_hidden_sizes=config['graph_embedding_net_settings']['edge_hidden_sizes'],
                                  node_hidden_sizes=config['graph_embedding_net_settings']['node_hidden_sizes'],
                                  n_prop_layers=config['graph_embedding_net_settings']['n_prop_layers'],
                                  share_prop_params=config['graph_embedding_net_settings']['share_prop_params'],
                                  edge_net_init_scale=config['graph_embedding_net_settings']['edge_net_init_scale'],
                                  node_update_type=config['graph_embedding_net_settings']['node_update_type'],
                                  use_reverse_direction=config['graph_embedding_net_settings']['use_reverse_direction'],
                                  reverse_dir_param_different=config['graph_embedding_net_settings'][
                                      'reverse_dir_param_different'],
                                  layer_norm=config['graph_embedding_net_settings']['layer_norm'],
                                  prop_type=config['graph_embedding_net_settings']['prop_type'])
    elif config['graph_embedding_net_settings']['prop_type'] == 'matching':
        # with cross-graph attention
        matching_aggregator = GraphAggregator_leaky(node_state_dim=config['aggregator_settings']['node_state_dim'],
                                                    node_hidden_sizes=config['aggregator_settings'][
                                                        'node_hidden_sizes'],
                                                    graph_transform_sizes=config['aggregator_settings'][
                                                        'graph_transform_sizes'],
                                                    gated=config['aggregator_settings']['gated'],
                                                    aggregation_type=config['aggregator_settings']['gated'])
        model = GraphMatchingNet(
            encoder, matching_aggregator, **config['graph_matching_net_settings'])
    else:
        raise ValueError('Unknown model type: %s' % config['training_settings']['embedding_or_matching'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['training_settings']['learning_rate'], weight_decay=1e-5)

    return model, optimizer


def build_yzd_datasets(config):
    config = copy.deepcopy(config)

    train_sample_fn_list = []
    vali_sample_fn_list = []
    training_set = YZDDataset(nm_dir=config['dataset_params']['nm_dir'],
                              am_dir=config['dataset_params']['am_dir'],
                              sample_fn_list=train_sample_fn_list,
                              num_epoch=config['training_settings']['num_epoch'],
                              inital_edge_feature_dim=config['dataset_params']['inital_edge_feature_dim'],
                              step_per_epoch=config['training_settings']['step_per_train_epoch'],
                              max_num_node_of_one_graph=config['dataset_params']['max_num_node_of_one_graph'])
    validation_set = YZDDataset(nm_dir=config['dataset_params']['nm_dir'],
                                am_dir=config['dataset_params']['am_dir'],
                                sample_fn_list=vali_sample_fn_list,
                                num_epoch=config['training_settings']['num_epoch'],
                                inital_edge_feature_dim=config['dataset_params']['inital_edge_feature_dim'],
                                step_per_epoch=config['training_settings']['step_per_vali_epoch'],
                                max_num_node_of_one_graph=config['dataset_params']['max_num_node_of_one_graph']
                                )
    return training_set, validation_set


def get_graph(batch):
    if len(batch) != 2:
        graph = batch
        node_features = torch.from_numpy(graph.node_features.astype('float32'))
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:
        graph, labels = batch
        node_features = torch.from_numpy(graph.node_features.astype('float32'))
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        labels = torch.from_numpy(labels).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels


def pack_datapoint(nm_list, am_list, edge_feature_dim=13):
    '''
    :param graphs: a list of (nm_matrix[num_nodes, node_feature_dims], am_matrix[num_edge, 2]) pairs. nm/am_matrixes are all numpy array.
    :return:
    '''
    num_node_list = [0]
    num_edge_list = []
    total_num_node = 0
    total_num_edge = 0
    batch_size = len(nm_list)
    for nm, am in zip(nm_list, am_list):
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
    edges = np.concatenate(am_list, axis=0)
    edges[..., 0] += scattered
    edges[..., 1] += scattered

    edge_features = np.zeros(shape=(total_num_edge, edge_feature_dim), dtype=np.float32)
    edge_features[np.arange(total_num_edge), edges[:, 2]] = 1

    return GraphData(from_idx=edges[..., 0],
                     to_idx=edges[..., 1],
                     node_features=node_features,
                     edge_features=edge_features,
                     graph_idx=np.repeat(np.arange(batch_size), np.array(num_node_list[1:])),
                     n_graphs=batch_size
                     )
