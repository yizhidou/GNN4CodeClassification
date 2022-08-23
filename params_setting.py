dataset_name = ''	# Juliet/Devign/ReVeal/POJ104
dataset_params = dict(dataset_name=dataset_name,
                      inital_node_feature_dim=71,
                      inital_edge_feature_dim=13,
                      max_num_node_of_one_graph=500,	# To filter out too large graphs
                      nm_dir='', 
                      am_dir='', # should be set based on dataset_name
                      train_vali_test_divide_record_path='' 
		      # If dataset_name=='Juliet', this paramter controls the specific dataset (juliet-multi/juliet-binary) you are using
)
training_settings = dict(batch_size=32,
                         learning_rate=1e-3,
                         pair_or_triplet_or_ce='ce',	# pair/triplet/ce loss
                         loss="margin",
                         margin=1.0,
                         graph_vec_regularizer_weight=1e-6,
                         clip_by_norm_or_by_value='norm',
                         value_correspond_with_clip=10.0,
                         num_epoch=100,
                         step_per_train_epoch=None,
                         step_per_vali_epoch=None,
                         if_decay=True,	# The flag on if lr schedular is used
                         decay_steps=1e6)
evaluation = dict(batch_size=32)
seed = 6
ckpt_save_dir = ''
training_log_dir = ''
gpu_idx = ''
encoder_settings = dict(node_feature_dim=dataset_params['inital_node_feature_dim'],
                        edge_feature_dim=dataset_params['inital_edge_feature_dim'],
                        node_hidden_sizes=[512, 256],
                        edge_hidden_sizes=[128, 64],
                        )
graph_embedding_net_settings = dict(node_state_dim=dataset_params['inital_node_feature_dim']
if not encoder_settings['node_hidden_sizes'] else
encoder_settings['node_hidden_sizes'][-1],
                                    edge_state_dim=dataset_params['inital_edge_feature_dim']
                                    if not encoder_settings['edge_hidden_sizes'] else
                                    encoder_settings['edge_hidden_sizes'][-1],
                                    node_hidden_sizes=[512, 256],
                                    edge_hidden_sizes=[13],
                                    n_prop_layers=3,	# the number of message-passing layers
                                    share_prop_params=False,
                                    edge_net_init_scale=0.1,
                                    node_update_type='gru',
                                    prop_type='embedding',	# two options is allowed: embedding/matching
				    # 'matching' means cross-graph attention is used; 'embedding' is the opposite
                                    use_reverse_direction=True,
                                    reverse_dir_param_different=True,
                                    layer_norm=True
                                    )
graph_matching_net_setting = graph_embedding_net_settings.copy()
graph_matching_net_setting['similarity'] = 'dotproduct'
aggregator_settings = dict(node_state_dim=encoder_settings['node_feature_dim']
if graph_embedding_net_settings['node_state_dim'] == None else
graph_embedding_net_settings['node_state_dim'],
                           node_hidden_sizes=[512, 256],
                           graph_transform_sizes=[512, 256, 16, 2],
                           gated=True,
                           aggregation_type="sum",
                           use_transformer=False	
			   # if True, then the last message-passing layer is replaced by a transformer encoder layer
)
