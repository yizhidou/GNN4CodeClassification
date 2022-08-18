from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
import numpy as np
import torch.nn as nn
import os
import datetime


def training(config, device, start_epoch_idx):
    training_set, validation_set = build_yzd_datasets(config)

    if config['training_settings']['pair_or_triplet_or_ce'] == 'pair':
        training_data_iter = training_set.pairs(config['training_settings']['batch_size'])
    elif config['training_settings']['pair_or_triplet_or_ce'] == 'triplet':
        training_data_iter = training_set.triplets(config['training_settings']['batch_size'])
    elif config['training_settings']['pair_or_triplet_or_ce'] == 'ce':
        training_data_iter = training_set.single(config['training_settings']['batch_size'])
        validation_singe_iter = validation_set.single(config['evaluation']['batch_size'])

    validation_pairs_iter = validation_set.pairs(batch_size=config['evaluation']['batch_size'])
    validation_triplet_iter = validation_set.triplets(batch_size=config['evaluation']['batch_size'])

    model, optimizer = build_model(config)
    model.to(device)

    if os.path.isfile(config['ckpt_save_path']):
        checkpoint = torch.load(config['ckpt_save_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('model reloaded from ckpt~')
    else:
        print('learning from scratch~')
    if config['training_settings']['if_decay']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=config['training_settings']['decay_steps'])

    training_n_graphs_in_batch = config['training_settings']['batch_size']
    if config['training_settings']['pair_or_triplet_or_ce'] == 'pair':
        training_n_graphs_in_batch *= 2
        step_per_train_epoch = config['training_settings']['step_per_train_epoch']
        step_per_vali_epoch = config['training_settings']['step_per_vali_epoch']
    elif config['training_settings']['pair_or_triplet_or_ce'] == 'triplet':
        training_n_graphs_in_batch *= 4
        step_per_train_epoch = config['training_settings']['step_per_train_epoch']
        step_per_vali_epoch = config['training_settings']['step_per_vali_epoch']
    elif config['training_settings']['pair_or_triplet_or_ce'] == 'ce':
        step_per_train_epoch = int(training_set.num_validate_sample / config['training_settings']['batch_size'])
        step_per_vali_epoch = int(validation_set.num_validate_sample / config['evaluation']['batch_size'])
    else:
        raise ValueError('Unknown training mode: %s' % config['training_settings']['pair_or_triplet_or_ce'])

    epoch_idx = start_epoch_idx
    info_str = ''
    epoch_loss_collection = []
    if config['training_settings']['pair_or_triplet_or_ce'] == 'pair':
        epoch_auc_collection = []
    elif config['training_settings']['pair_or_triplet_or_ce'] == 'triplet':
        epoch_acc_collection = []
    elif config['training_settings']['pair_or_triplet_or_ce'] == 'ce':
        epoch_acc_collection = []
    time_base = datetime.datetime.now()
    for train_batch_idx, train_batch in enumerate(training_data_iter):
        batch_idx_of_epoch = train_batch_idx - (epoch_idx - start_epoch_idx) * step_per_train_epoch
        optimizer.zero_grad()
        model.train(mode=True)
        if config['training_settings']['pair_or_triplet_or_ce'] == 'pair':
            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(train_batch)
            labels = labels.to(device)
            _, graph_vectors = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                     to_idx.to(device),
                                     graph_idx.to(device), training_n_graphs_in_batch)
            x, y = reshape_and_split_tensor(graph_vectors, 2)
            loss = pairwise_loss(x, y, labels,
                                 loss_type=config['training_settings']['loss'],
                                 margin=config['training_settings']['margin'])

            is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
            is_neg = 1 - is_pos
            n_pos = torch.sum(is_pos)
            n_neg = torch.sum(is_neg)
            sim = compute_similarity(config, x, y)

            sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
            sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)
            # print('the size of is_pos = {}; the size of is_neg = {}'.format(list(is_pos.size()), list(is_neg.size())))

            # similarity_train = compute_similarity(config, x, y)
            pair_auc_train = auc(sim, labels)
            sim_diff = sim_pos - sim_neg
            batch_mean_loss = torch.mean(loss)
            epoch_loss_collection.append(batch_mean_loss)

            epoch_auc_collection.append(pair_auc_train)

            graph_vec_scale = torch.mean(graph_vectors ** 2)
            if config['training_settings']['graph_vec_regularizer_weight'] > 0:
                loss = loss + (config['training_settings']['graph_vec_regularizer_weight'] *
                               0.5 * graph_vec_scale)

            loss.backward(torch.ones_like(loss))  #

            new_info = 'batch{}_epoch_{}: batch_mean_loss = {}; sim_pos = {}; sim_neg = {}; sim_diff = {}; pair_auc(train) = {}\n'. \
                format(batch_idx_of_epoch,
                       epoch_idx, batch_mean_loss.cpu().detach().numpy().item(), sim_pos, sim_neg, sim_diff,
                       pair_auc_train)
            print(new_info)
            info_str += new_info
            if config['training_settings']['clip_by_norm_or_by_value'] == 'value':
                nn.utils.clip_grad_value_(model.parameters(), config['training_settings']['value_correspond_with_clip'])
            elif config['training_settings']['clip_by_norm_or_by_value'] == 'norm':
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=config['training_settings']['value_correspond_with_clip'])
            optimizer.step()

        elif config['training_settings']['pair_or_triplet_or_ce'] == 'triplet':
            node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(train_batch)

            # print('shape of node_features is {}'.format(node_features.shape))
            _, graph_vectors = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                     to_idx.to(device),
                                     graph_idx.to(device), training_n_graphs_in_batch)
            x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
            loss = triplet_loss(x_1, y, x_2, z,
                                loss_type=config['training_settings']['loss'],
                                margin=config['training_settings']['margin'])
            sim1_train = compute_similarity(config, x_1, y)
            # print('x_1 = {}; y = {}'.format(x_1, y))
            sim2_train = compute_similarity(config, x_2, z)
            # print('x_2 = {}; y = {}'.format(x_2, y))

            sim_pos = torch.mean(compute_similarity(config, x_1, y))
            sim_neg = torch.mean(compute_similarity(config, x_2, z))
            triplet_acc_train = torch.mean((sim1_train > sim2_train).float())
            sim_diff = sim_pos - sim_neg
            batch_mean_loss = torch.mean(loss)
            epoch_loss_collection.append(batch_mean_loss)
            epoch_acc_collection.append(triplet_acc_train)

            graph_vec_scale = torch.mean(graph_vectors ** 2)
            if config['training_settings']['graph_vec_regularizer_weight'] > 0:
                loss = loss + (config['training_settings']['graph_vec_regularizer_weight'] *
                               0.5 * graph_vec_scale)

            loss.backward(torch.ones_like(loss))  #

            new_info = 'batch{}_epoch_{}: batch_mean_loss = {}; sim_pos = {}; sim_neg = {}; sim_diff = {}; triplet_acc(train) = {}\n'. \
                format(batch_idx_of_epoch, epoch_idx, batch_mean_loss.cpu().detach().numpy().item(), sim_pos, sim_neg,
                       sim_diff, triplet_acc_train)
            print(new_info)
            info_str += new_info

            if config['training_settings']['clip_by_norm_or_by_value'] == 'value':
                nn.utils.clip_grad_value_(model.parameters(), config['training_settings']['value_correspond_with_clip'])
            elif config['training_settings']['clip_by_norm_or_by_value'] == 'norm':
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=config['training_settings']['value_correspond_with_clip'])
            optimizer.step()
        else:  # 'ce' case
            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(train_batch)
            labels = labels.to(device)
            _, model_predicted_label_batch = model(node_features.to(device), edge_features.to(device),
                                                   from_idx.to(device),
                                                   to_idx.to(device),
                                                   graph_idx.to(device), training_n_graphs_in_batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(model_predicted_label_batch, labels)
            loss.backward()
            epoch_loss_collection.append(loss)
            model_predicted_label_batch_argmax = torch.argmax(model_predicted_label_batch, 1)
            correct_batch = (model_predicted_label_batch_argmax == labels).sum().item()
            batch_acc = float(correct_batch) / float(config['training_settings']['batch_size'])
            epoch_acc_collection.append(batch_acc)

            new_info = 'batch{}_epoch_{}: batch_loss = {}; batch_acc = {}; model_predicted_label_batch = {}; label_batch = {}\n'. \
                format(batch_idx_of_epoch, epoch_idx, loss.cpu().detach().numpy().item(), batch_acc,
                       model_predicted_label_batch_argmax.cpu().detach().numpy(), labels.cpu().detach().numpy())
            print(new_info)
            info_str += new_info

            if config['training_settings']['clip_by_norm_or_by_value'] == 'value':
                nn.utils.clip_grad_value_(model.parameters(), config['training_settings']['value_correspond_with_clip'])
            elif config['training_settings']['clip_by_norm_or_by_value'] == 'norm':
                nn.utils.clip_grad_norm_(model.parameters(),
                                         max_norm=config['training_settings']['value_correspond_with_clip'])
            optimizer.step()

        if (train_batch_idx + 1) % 50 == 0:  # dump to log every 50 batch
            with open(config['training_log_path'], 'a') as info_logger:
                info_logger.write(info_str)
                info_str = ''
        if (train_batch_idx + 1) % step_per_train_epoch == 0:
            if config['training_settings']['pair_or_triplet_or_ce'] == 'pair':
                epoch_avg_auc = sum(epoch_auc_collection) / len(epoch_auc_collection)
                epoch_auc_collection = []
                epoch_avg_loss = sum(epoch_loss_collection) / len(epoch_loss_collection)
                epoch_loss_collection = []
                new_info = 'epoch_{}_training(pair): avg_loss = {}; avg_auc = {}\n'.format(epoch_idx, epoch_avg_loss,
                                                                                           epoch_avg_auc)

            elif config['training_settings']['pair_or_triplet_or_ce'] == 'triplet':
                epoch_avg_acc = sum(epoch_acc_collection) / len(epoch_acc_collection)
                epoch_acc_collection = []
                epoch_avg_loss = sum(epoch_loss_collection) / len(epoch_loss_collection)
                epoch_loss_collection = []
                new_info = 'epoch_{}_training(triplet): avg_loss = {}; avg_acc = {}\n'.format(epoch_idx, epoch_avg_loss,
                                                                                              epoch_avg_acc)

            else:  # ce case
                epoch_accumulated_loss = sum(epoch_loss_collection)
                epoch_loss_collection = []
                epoch_avg_acc = sum(epoch_acc_collection) / len(epoch_acc_collection)
                epoch_acc_collection = []
                new_info = 'epoch_{}_training(ce): epoch_loss = {}; avg_acc = {}\n'.format(epoch_idx,
                                                                                           epoch_accumulated_loss,
                                                                                           epoch_avg_acc)

            new_info += 'epoch_{} train time_interval: {}\n'.format(epoch_idx, datetime.datetime.now() - time_base)
            print(new_info)
            info_str += new_info
            model.eval()
            with torch.no_grad():
                if config['training_settings']['pair_or_triplet_or_ce'] == 'pair' or config['training_settings'][
                    'pair_or_triplet_or_ce'] == 'triplet':
                    accumulated_pair_auc = []
                    for vali_pair_batch_idx in range(step_per_vali_epoch):
                        batch = next(validation_pairs_iter)
                        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                        labels = labels.to(device)
                        _, eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
                                              to_idx.to(device),
                                              graph_idx.to(device), config['evaluation']['batch_size'] * 2)

                        x, y = reshape_and_split_tensor(eval_pairs, 2)
                        similarity = compute_similarity(config, x, y)
                        pair_auc = auc(similarity, labels)
                        accumulated_pair_auc.append(pair_auc)
                        new_info = 'batch_{}_of_validation_epoch_{}(pair): pair_auc = {}\n'.format(
                            vali_pair_batch_idx,
                            epoch_idx,
                            pair_auc)
                        print(new_info)
                        info_str += new_info
                        if (vali_pair_batch_idx + 1) % 50 == 0:
                            with open(config['training_log_path'], 'a') as info_logger:
                                info_logger.write(info_str)
                                info_str = ''

                    accumulated_triplet_acc = []
                    for vali_triplet_batch_idx in range(step_per_vali_epoch):
                        batch = next(validation_triplet_iter)
                        node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch)
                        _, eval_triplets = model(node_features.to(device), edge_features.to(device),
                                                 from_idx.to(device),
                                                 to_idx.to(device),
                                                 graph_idx.to(device),
                                                 config['evaluation']['batch_size'] * 4)
                        x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
                        # print('x1 = {}\n x2 = {}'.format(x_1, x_2))
                        sim_1 = compute_similarity(config, x_1, y)
                        sim_2 = compute_similarity(config, x_2, z)
                        # print('sim_1(triplet) = {}; sim_2(triplet) = {}'.format(sim_1, sim_2))
                        triplet_acc = torch.mean((sim_1 > sim_2).float())
                        accumulated_triplet_acc.append(triplet_acc.cpu().numpy())
                        new_info = 'batch_{}_of_validation_epoch_{}(triplet): triplet_acc = {}\n'.format(
                            vali_triplet_batch_idx, epoch_idx, triplet_acc)
                        print(new_info)
                        info_str += new_info
                        if (vali_triplet_batch_idx + 1) % 50 == 0:
                            with open(config['training_log_path'], 'a') as info_logger:
                                info_logger.write(info_str)
                                info_str = ''
                    info_str += 'validation_epoch_{}: mean_accumulated_pair_auc = {}; mean_accumulated_triplet_acc = {}\n'.format(
                        epoch_idx,
                        np.mean(accumulated_pair_auc),
                        np.mean(accumulated_triplet_acc))
                else:
                    accumulated_acc = []
                    for vali_batch_idx in range(step_per_vali_epoch):
                        batch = next(validation_singe_iter)
                        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                        labels = labels.to(device)
                        _, model_predicted_label_batch = model(node_features.to(device), edge_features.to(device),
                                                               from_idx.to(device),
                                                               to_idx.to(device),
                                                               graph_idx.to(device), config['evaluation']['batch_size'])
                        model_predicted_label_batch_argmax = torch.argmax(model_predicted_label_batch, 1)
                        correct_batch = (model_predicted_label_batch_argmax == labels).sum().item()
                        batch_acc = float(correct_batch) / float(config['evaluation']['batch_size'])

                        accumulated_acc.append(batch_acc)
                        new_info = 'batch_{}_of_validation_epoch_{}(ce): batch_acc = {}; model_pred_label = {}; labels = {}\n'.format(
                            vali_batch_idx,
                            epoch_idx,
                            batch_acc,
                            model_predicted_label_batch_argmax.cpu().detach().numpy(),
                            labels.cpu().detach().numpy())
                        print(new_info)
                        info_str += new_info
                        if (vali_batch_idx + 1) % 50 == 0:
                            with open(config['training_log_path'], 'a') as info_logger:
                                info_logger.write(info_str)
                                info_str = ''
                    info_str += 'validation_epoch_{}: mean_validation_acc = {};\n'.format(
                        epoch_idx,
                        np.mean(accumulated_acc))

                with open(config['training_log_path'], 'a') as info_logger:
                    info_logger.write(info_str)
                    info_str = ''
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, config['ckpt_save_path'])
                print('model saved~')
            model.train()
            if config['training_settings']['if_decay']:
                scheduler.step()
            epoch_idx += 1
            time_base = datetime.datetime.now()

    with open(config['training_log_path'], 'a') as info_logger:
        info_logger.write(info_str)


def parse_params() -> dict:
    '''
    load hyper-params from separate config file (./params_setting.py)
    '''
    pass
