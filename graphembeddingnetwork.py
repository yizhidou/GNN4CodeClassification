import torch
import torch.nn as nn
from segment import unsorted_segment_sum
from copy import deepcopy


class GraphEncoder(nn.Module):
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self,
                 node_feature_dim,
                 edge_feature_dim,
                 node_hidden_sizes=None,
                 edge_hidden_sizes=None,
                 name='graph-encoder'):
        """Constructor.

        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(GraphEncoder, self).__init__()

        # this also handles the case of an empty list
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes
        self._build_model()

    def _build_model(self):
        if self._node_hidden_sizes is not None and len(self._node_hidden_sizes) > 0:
            layer = []
            layer.append(nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP1 = nn.Sequential(*layer)

        if self._edge_hidden_sizes is not None and len(self._edge_hidden_sizes) > 0:
            layer = []
            layer.append(nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._edge_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP2 = nn.Sequential(*layer)
        else:
            self.MLP2 = None

    def forward(self, node_features, edge_features=None):
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if self._node_hidden_sizes is None or len(self._node_hidden_sizes) == 0:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
        if edge_features is None or self._edge_hidden_sizes is None or len(self._edge_hidden_sizes) == 0:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(edge_features)  # original code was wrong!!!

        return node_outputs, edge_outputs


def graph_prop_once(node_states,
                    from_idx,
                    to_idx,
                    message_net,
                    aggregation_module=None,
                    edge_features=None):
    """One round of propagation (message passing) in a graph.

    Args:
      node_states: [n_nodes, node_state_dim] float tensor, node state vectors, one
        row for each node.
      from_idx: [n_edges] int tensor, index of the from nodes.
      to_idx: [n_edges] int tensor, index of the to nodes.
      message_net: a network that maps concatenated edge inputs to message
        vectors.
      aggregation_module: a module that aggregates messages on edges to aggregated
        messages for each node.  Should be a callable and can be called like the
        following,
        `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
        where messages is [n_edges, edge_message_dim] tensor, to_idx is the index
        of the to nodes, i.e. where each message should go to, and n_nodes is an
        int which is the number of nodes to aggregate into.
      edge_features: if provided, should be a [n_edges, edge_feature_dim] float
        tensor, extra features for each edge.

    Returns:
      aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
        aggregated messages, one row for each node.
    """
    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]

    if edge_features is not None:
        edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)
    messages = message_net(edge_inputs)

    from segment import unsorted_segment_sum

    tensor = unsorted_segment_sum(messages, to_idx, node_states.shape[0])
    return tensor


class GraphPropLayer(nn.Module):
    """Implementation of a graph propagation (message passing) layer."""

    def __init__(self,
                 node_state_dim,
                 edge_hidden_sizes,  # int
                 node_hidden_sizes,  # int
                 prop_type,
                 node_update_type,
                 edge_state_dim=None,  # added by yzd
                 edge_net_init_scale=0.1,
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 name='graph-net'):
        """Constructor.

        Args:
          node_state_dim: int, dimensionality of node states.
          edge_hidden_sizes: list of ints, hidden sizes for the edge message
            net, the last element in the list is the size of the message vectors.
          node_hidden_sizes: list of ints, hidden sizes for the node update
            net.
          edge_net_init_scale: initialization scale for the edge networks.  This
            is typically set to a small value such that the gradient does not blow
            up.
          node_update_type: type of node updates, one of {mlp, gru, residual}.
          use_reverse_direction: set to True to also propagate messages in the
            reverse direction.
          reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.
          layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphPropLayer, self).__init__()

        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]

        # output size is node_state_dim
        # print('~~~~~~~~~~~~~~~ node_hidden_sizes = {}'.format(node_hidden_sizes))
        if not node_hidden_sizes:
            self._node_hidden_sizes = [node_state_dim]
        else:
            self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type

        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different

        self._layer_norm = layer_norm
        self._prop_type = prop_type
        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm(self._edge_hidden_sizes[-1])
            self.layer_norm2 = nn.LayerNorm(self._node_hidden_sizes[-1])

    def build_model(self):
        layer = []
        if self._edge_state_dim is None:
            layer.append(nn.Linear(2 * self._node_state_dim, self._edge_hidden_sizes[0]))  # here was wrong too...
        else:
            layer.append(nn.Linear(2 * self._node_state_dim + self._edge_state_dim, self._edge_hidden_sizes[0]))
        # print('in gnn build model, edge_hidden_sizes = {}'.format(self._edge_hidden_sizes))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layer)
        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                if self._edge_state_dim is None:
                    layer.append(nn.Linear(2 * self._node_state_dim, self._edge_hidden_sizes[-1]))
                else:
                    # print('edge_hidden_sizes = {}'.format(self._edge_hidden_sizes))
                    layer.append(
                        nn.Linear(2 * self._node_state_dim + self._edge_state_dim, self._edge_hidden_sizes[-1]))
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net
        # print('prop_type = {}'.format(self._prop_type))
        if self._node_update_type == 'gru':
            if self._prop_type == 'embedding':
                # print('_node_state_dim = {}; edge_state_dim = {}'.format(self._node_state_dim, self._edge_state_dim))
                self.GRU = torch.nn.GRU(self._node_state_dim + self._edge_hidden_sizes[-1],
                                        self._node_state_dim)  # this is the worked version
                # self.GRU = torch.nn.GRU(self._node_state_dim * 2, self._node_state_dim)
            elif self._prop_type == 'matching':
                # self.GRU = torch.nn.GRU(self._node_state_dim * 3, self._node_state_dim)   # why *3 ?
                # print('in defining gru, the _node_state_dim is expected to be: {}'.format(self._node_state_dim))
                self.GRU = torch.nn.GRU(self._edge_hidden_sizes[-1] + self._node_state_dim * 2, self._node_state_dim)
                # self.GRU = torch.nn.GRU(self._node_state_dim * 3, self._node_state_dim)     # this is the worked version
        else:
            layer = []
            if self._prop_type == 'embedding':
                layer.append(
                    nn.Linear(self._node_hidden_sizes[0] + self._edge_hidden_sizes[-1], self._node_hidden_sizes[1]))
            elif self._prop_type == 'matching':
                layer.append(nn.Linear(self._node_state_dim * 4, self._node_hidden_sizes[0]))
            for i in range(2, len(self._node_hidden_sizes)):
                # print('here self._node_hidden_sizes = {}'.format(self._node_hidden_sizes))
                layer.append(nn.ReLU())
                linear = nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i])
                if self._node_update_type == 'residual' and i == len(self._node_hidden_sizes) - 1:
                    torch.nn.init.zeros_(linear.weight)
                layer.append(linear)
            self.MLP = nn.Sequential(*layer)

    def _compute_aggregated_messages(
            self, node_states, from_idx, to_idx, edge_features=None):
        """Compute aggregated messages for each node.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.

        Returns:
          aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
            aggregated messages for each node.
        """
        aggregated_messages = graph_prop_once(
            node_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            edge_features=edge_features)
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(
                node_states,
                to_idx,
                from_idx,
                self._reverse_message_net,
                aggregation_module=None,
                edge_features=edge_features)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)
        return aggregated_messages

    def _compute_node_update(self,
                             node_states,
                             node_state_inputs,
                             node_features=None):
        """Compute node updates.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, the input node
            states.
          node_state_inputs: a list of tensors used to compute node updates.  Each
            element tensor should have shape [n_nodes, feat_dim], where feat_dim can
            be different.  These tensors will be concatenated along the feature
            dimension.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          new_node_states: [n_nodes, node_state_dim] float tensor, the new node
            state tensor.

        Raises:
          ValueError: if node update type is not supported.
        """
        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            # print('here shape of node_state_inputs = {}'.format(node_state_inputs.shape))
            mlp_output = self.MLP(node_state_inputs)
            if self._layer_norm:
                mlp_output = self.layer_norm2(mlp_output)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                # print('if you see me, then we are in residual mode~~~')
                return node_states + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                edge_features=None,
                node_features=None):
        """Run one propagation step.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.
        """
        # print('In GraphProbLayer, shape of node_states = {}'.format(list(node_states.size())))
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        return self._compute_node_update(node_states=node_states,
                                         node_state_inputs=[aggregated_messages],
                                         node_features=node_features)


class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 node_state_dim,
                 node_hidden_sizes,
                 use_transformer,
                 use_mask,
                 graph_transform_sizes=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        # print('before gate, node_hidden_sizes = {}'.format(node_hidden_sizes))
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = node_state_dim
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()
        self.use_transformer = use_transformer
        self.use_mask = use_mask
        if self.use_transformer:
            self.TransformerLayer = torch.nn.TransformerEncoderLayer(d_model=self._input_size, nhead=4,
                                                                     dim_feedforward=4 * self._input_size,
                                                                     batch_first=True,
                                                                     dropout=0)
            self.device = torch.device('cuda:0')

    def build_model(self):
        node_hidden_sizes = deepcopy(self._node_hidden_sizes)
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        # print('~~~~~~~~~~~~~~~~ self._input_size = {}, node_hidden_sizes[0] = {}'.format(self._input_size,
        #                                                                                  node_hidden_sizes[0]))

        # print('gate node_hidden_sizes = {}'.format(node_hidden_sizes))
        # print('shape in MLP1:')
        # print('{} {}'.format(self._input_size, node_hidden_sizes[0]))
        layer.append(nn.Linear(self._input_size, node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
            # print('{}, {}'.format(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0:
            layer = []
            # print('shape in MLP2:')
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            # print('{}, {}'.format(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
                # print('{}, {}'.format(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)
        else:
            MLP2 = None

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx: [n_nodes] int tensor, graph ID for each node.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """
        if self.use_transformer:
            # graph_idx: [total_num_nodes, ]    (0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3)
            # n_graphs = 4
            ones = torch.ones_like(graph_idx, device=self.device)
            # [total_num_nodes, ]   (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
            num_nodes = torch.zeros(n_graphs, dtype=ones.dtype, device=self.device)
            # [n_graphs, ]  (0, 0, 0, 0)
            num_nodes.scatter_add_(0, graph_idx, ones)
            # [n_graphs, ] (2, 3, 2, 4)
            cumsum_num_nodes = torch.cat((torch.zeros(1, dtype=num_nodes.dtype, device=self.device),
                                          torch.cumsum(num_nodes, 0)[:-1]))
            # [n_graphs, ] (0, 2, 5, 7)
            replicated_cumsum_num_nodes = torch.index_select(cumsum_num_nodes, 0, graph_idx)
            # [total_num_nodes, ]   (0, 0, 2, 2, 2, 5, 5, 7, 7, 7, 7)
            total_num_nodes = graph_idx.shape[0]
            node_idx = torch.arange(total_num_nodes, dtype=torch.int, device=self.device) - replicated_cumsum_num_nodes
            # [total_num_nodes, ]   (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) - (0, 0, 2, 2, 2, 5, 5, 7, 7, 7, 7)
            # (0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3)

            dst = torch.zeros(n_graphs * 500, self._input_size, device=self.device)
            # [4*500, 256]
            write_idx = node_idx + graph_idx * 500
            # [total_num_nodes, ]   (0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3) + (0*500, 0*500, 1*500, 1*500, 1*500, 2*500, 2*500, 3*500, 3*500, 3*500, 3*500)
            try:
                padded_node_states = dst.index_add(0, write_idx, node_states)
                print('dst.shape = {}; write_idx.shape = {}; node_states.shape = {}'.format(dst.shape,
                                                                                            write_idx.shape,
                                                                                            node_states.shape))
            except RuntimeError:
                print('runtime error! dst.shape = {}; write_idx.shape = {}; node_states.shape = {}'.format(dst.shape,
                                                                                                           write_idx.shape,
                                                                                                           node_states.shape))
                exit(666)
            padded_node_states = torch.reshape(padded_node_states, (n_graphs, 500, self._input_size))
            print('the shape of padded_node_states = {}'.format(padded_node_states.shape))
            if self.use_mask:
                mask = torch.ones(size=(n_graphs * 500,), device=self.device).scatter_(dim=0, index=write_idx,
                                                                                       src=torch.zeros(
                                                                                           (n_graphs * 500,),
                                                                                           device=self.device)).reshape(
                    n_graphs, 500)
                # mask = torch.ones(size=(n_graphs * 500,), device=self.device).index_add(dim=0, index=write_idx,
                #                                                                         source=torch.ones((n_graphs * 500,),
                #                                                                                           device=self.device),
                #                                                                         alpha=-1).reshape(n_graphs, 500)
                transformed_node_states = self.TransformerLayer(src=padded_node_states, src_key_padding_mask=mask)
            else:
                transformed_node_states = self.TransformerLayer(src=padded_node_states)
            transformed_node_states = transformed_node_states.reshape(n_graphs * 500, self._graph_state_dim)
            node_states = torch.index_select(transformed_node_states, dim=0, index=write_idx)
            # print('total_num_nodes = {}'.format(total_num_nodes))
            # print('after transformer, shape of nodes_states = {}'.format(node_states.shape))
            # exit(666)

        # print('in aggregator, before MLP1, embedding is (shape={}):'.format(node_states.shape))
        # print(node_states)
        node_states_g = self.MLP1(node_states)
        # print('In agg forward, the shape of node_states_g = {}'.format(list(node_states_g.size())))
        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            # print('the shape of a is: {}; the shape of b is: {}'.format(node_states_g[..., self._graph_state_dim:].shape,
            #                                                             gates.shape))
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        # print('In agg forward, node_states_g.shape = {}; graph_idx.shape = {}; n_graphs={}; '.format(node_states_g.shape,
        #                                                                                           graph_idx.shape,
        #                                                                                           n_graphs))
        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)
        # print('In agg forward, the shape of graph_states = {}'.format(list(graph_states.size())))
        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)
        # transform the reduced graph states further

        # print('in graphembeddingnet, before mlp2, graph_sates = {}'.format(graph_states.cpu().detach().numpy()))
        graph_embedding_1 = None
        if self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0:
            # print('in graphembeddingnet, the third from last, graph_sates = {}'.format(
            #     self.MLP2[:-2](graph_states).cpu().detach().numpy()))
            # print('in graphembeddingnet, the second from last, graph_sates = {}'.format(
            #     self.MLP2[:-1](graph_states).cpu().detach().numpy()))
            graph_embedding_1 = self.MLP2[:-1](graph_states)
            graph_states = self.MLP2(graph_states)
        # print('in graphembeddingnet, after mlp2, graph_sates = {}'.format(graph_states.cpu().detach().numpy()))
        return graph_embedding_1, graph_states

class GraphAggregator_leaky(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 node_state_dim,
                 node_hidden_sizes,
                 graph_transform_sizes=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator_leaky, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        # print('before gate, node_hidden_sizes = {}'.format(node_hidden_sizes))
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = node_state_dim
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = deepcopy(self._node_hidden_sizes)
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        # print('~~~~~~~~~~~~~~~~ self._input_size = {}, node_hidden_sizes[0] = {}'.format(self._input_size,
        #                                                                                  node_hidden_sizes[0]))

        # print('gate node_hidden_sizes = {}'.format(node_hidden_sizes))
        # print('shape in MLP1:')
        # print('{} {}'.format(self._input_size, node_hidden_sizes[0]))
        layer.append(nn.Linear(self._input_size, node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
            # print('{}, {}'.format(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0:
            layer = []
            # print('shape in MLP2:')
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            # print('{}, {}'.format(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.LeakyReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
                # print('{}, {}'.format(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)
        else:
            MLP2 = None

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx: [n_nodes] int tensor, graph ID for each node.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """

        node_states_g = self.MLP1(node_states)
        # print('In agg forward, the shape of node_states_g = {}'.format(list(node_states_g.size())))
        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            # print('the shape of a is: {}; the shape of b is: {}'.format(node_states_g[..., self._graph_state_dim:].shape,
            #                                                             gates.shape))
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        # print('In agg forward, node_states_g.shape = {}; graph_idx.shape = {}; n_graphs={}; '.format(node_states_g.shape,
        #                                                                                           graph_idx.shape,
        #                                                                                           n_graphs))
        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)
        # print('In agg forward, the shape of graph_states = {}'.format(list(graph_states.size())))
        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)
        # transform the reduced graph states further

        # print('in graphembeddingnet, before mlp2, graph_sates = {}'.format(graph_states.cpu().detach().numpy()))
        graph_embedding_1 = None
        if self._graph_transform_sizes is not None and len(self._graph_transform_sizes) > 0:
            # print('in graphembeddingnet, the third from last, graph_sates = {}'.format(
            #     self.MLP2[:-2](graph_states).cpu().detach().numpy()))
            # print('in graphembeddingnet, the second from last, graph_sates = {}'.format(
            #     self.MLP2[:-1](graph_states).cpu().detach().numpy()))

            graph_embedding_1 = self.MLP2[:-1](graph_states)
            graph_states = self.MLP2(graph_states)
        # print('in graphembeddingnet, after mlp2, graph_sates = {}'.format(graph_states.cpu().detach().numpy()))
        return graph_embedding_1, graph_states


class GraphEmbeddingNet(nn.Module):
    """A graph to embedding mapping network."""

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 prop_type,
                 node_update_type,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 name='graph-embedding-net'):
        """Constructor.

        Args:
          encoder: GraphEncoder, encoder that maps features to embeddings.
          aggregator: GraphAggregator, aggregator that produces graph
            representations.

          node_state_dim: dimensionality of node states.
          edge_hidden_sizes: sizes of the hidden layers of the edge message nets.
          node_hidden_sizes: sizes of the hidden layers of the node update nets.

          n_prop_layers: number of graph propagation layers.

          share_prop_params: set to True to share propagation parameters across all
            graph propagation layers, False not to.
          edge_net_init_scale: scale of initialization for the edge message nets.
          node_update_type: type of node updates, one of {mlp, gru, residual}.
          use_reverse_direction: set to True to also propagate messages in the
            reverse direction.
          reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.

          layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphEmbeddingNet, self).__init__()

        self._encoder = encoder
        self._aggregator = aggregator
        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        # print('in graphembedding net, node_hidden_sizes = {}'.format(node_hidden_sizes))
        self._n_prop_layers = n_prop_layers
        self._share_prop_params = share_prop_params
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._layer_norm = layer_norm
        self._prop_layers = []
        self._prop_layers = nn.ModuleList()
        self._layer_class = layer_class
        self._prop_type = prop_type
        self.build_model()

    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            node_state_dim=self._node_state_dim,
            edge_state_dim=self._edge_state_dim,
            edge_hidden_sizes=self._edge_hidden_sizes,
            node_hidden_sizes=self._node_hidden_sizes,
            edge_net_init_scale=self._edge_net_init_scale,
            node_update_type=self._node_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm,
            prop_type=self._prop_type)
        # name='graph-prop-%d' % layer_id)

    def build_model(self):
        if len(self._prop_layers) < self._n_prop_layers:
            # build the layers
            for i in range(self._n_prop_layers):
                if i == 0 or not self._share_prop_params:
                    layer = self._build_layer(i)

                else:
                    layer = self._prop_layers[0]
                self._prop_layers.append(layer)
                # print('layer_{} produced'.format(i))

    def _apply_layer(self,
                     layer,
                     node_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     node_features,
                     edge_features):
        """Apply one layer on the given inputs."""
        del graph_idx, n_graphs
        return layer(node_states, from_idx, to_idx, node_features=node_features, edge_features=edge_features)

    def forward(self,
                node_features,
                edge_features,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs):
        """Compute graph representations.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: [n_edges, edge_feat_dim] float tensor.
          from_idx: [n_edges] int tensor, index of the from node for each edge.
          to_idx: [n_edges] int tensor, index of the to node for each edge.
          graph_idx: [n_nodes] int tensor, graph id for each node.
          n_graphs: int, number of graphs in the batch.

        Returns:
          graph_representations: [n_graphs, graph_representation_dim] float tensor,
            graph representations.
        """
        # print('before encoder, node_states.shape = {}'.format(node_features.shape))
        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features
        # print('after encoder, node_features.shape = {}; edge_features.shape = {}'.format(list(node_features.size()),
        #                                                                                  list(edge_features.size())))

        # layer_outputs = [node_states]
        # print('after encoder, before gnn, node_states.shape = {}'.format(node_states.shape))
        for layer_idx, layer in enumerate(self._prop_layers):
            node_states = self._apply_layer(
                layer,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                node_features,
                edge_features)
        # print('after gnn, node_states.shape = {}'.format(node_states.shape))
        # layer_outputs.append(node_states)

        # these tensors may be used e.g. for visualization
        # self._layer_outputs = layer_outputs
        graph_states, final_states = self._aggregator(node_states, graph_idx, n_graphs)
        # print('after aggregator, result = {}'.format(list(result.size())))
        return graph_states, final_states

    def reset_n_prop_layers(self, n_prop_layers):
        """Set n_prop_layers to the provided new value.

        This allows us to train with certain number of propagation layers and
        evaluate with a different number of propagation layers.

        This only works if n_prop_layers is smaller than the number used for
        training, or when share_prop_params is set to True, in which case this can
        be arbitrarily large.

        Args:
          n_prop_layers: the new number of propagation layers to set.
        """
        self._n_prop_layers = n_prop_layers

    @property
    def n_prop_layers(self):
        return self._n_prop_layers

    def get_layer_outputs(self):
        """Get the outputs at each layer."""
        if hasattr(self, '_layer_outputs'):
            return self._layer_outputs
        else:
            raise ValueError('No layer outputs available.')
