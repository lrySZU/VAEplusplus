import argparse
import random
import os
import time
from datetime import datetime
import ast
import numpy as np
import torch
from mymodel import VAE
from mydataset import load_targetdata, load_auxiliary_data, loadTestData
import mytest
from scipy import sparse
import logging
import matplotlib.pyplot as plt


def setup_seed(seed):
    """
    the random seed scope is from the time of setting to the next time it is set
    To replicate the results, set the same random seed
    :param seed:
    :return:
    """
    torch.manual_seed(seed)  # CPU seed, return a torch.Generator
    torch.cuda.manual_seed(seed)  # GPU seed, return a torch.Generator
    np.random.seed(seed)
    random.seed(seed)
    torch.are_deterministic_algorithms_enabled()  # only use deterministic convolution algorithms


def parser_args():
    """
    explanation of command-line options, arguments, and subcommands
    :return:
    """
    # 1. Create interpreter
    parser = argparse.ArgumentParser()

    # 2. add the required parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--model', type=str, default='MultiVAE')
    parser.add_argument('--lr-rate', type=float, default=1e-3)
    parser.add_argument('--reg-scale', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('-hiddenDim', nargs='+', type=int, default=100)
    # training parameters
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--path', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='Rec15')
    parser.add_argument('--transaction', type=str, default='target_train')
    parser.add_argument('--examination', type=str, default='auxiliary')
    parser.add_argument('--test', type=str, default='target_test')
    parser.add_argument('--user_num', type=int, default=36917)
    parser.add_argument('--item_num', type=int, default=9621)
    # other parameters
    parser.add_argument('--total_anneal_steps', type=int, default=200000)
    parser.add_argument('--is_train', type=ast.literal_eval, default=True)

    return parser.parse_args()


def data_visualization(recall_list, precision_list, ndcg_list, length):
    epochs = [10 * i for i in range(1, length + 1)]
    fig, ax = plt.subplots()
    plt.plot(epochs, recall_list, color='r', label='recall@5')
    plt.plot(epochs, precision_list, color='b', label='precision@5')
    plt.plot(epochs, ndcg_list, color='g', label="NDCG@5")
    plt.legend()
    plt.xlabel("iterate itme")
    plt.ylabel('metrics')
    plt.title('metrics of VAE++')
    plt.show()


def train(args, criterion, model: VAE, adam_optimizer, targetData_matrix, auxiliaryData_matrix,
          userList_train, testDict, userList_test, topN=[5, 10, 15, 20]):
    stop_count, anneal_cap, update_count = 0, 0.2, 0.0
    best_prec5, best_rec5, best_f15, best_ndcg5, best_1call5, best_iter = 0., 0., 0., 0., 0., 0
    recall_list = []
    precision_list = []
    ndcg_list = []
    for epoch in range(args.epoch):
        random.shuffle(userList_train)

        loss = 0.
        for batch_num, sta_idx in enumerate(range(0, len(userList_train), args.batch_size)):
            end_idx = min(sta_idx + args.batch_size, len(userList_train))
            batch_index = userList_train[sta_idx: end_idx]

            if args.total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = anneal_cap

            A = auxiliaryData_matrix[batch_index]
            if sparse.isspmatrix(A):
                A = A.toarray()

            T = targetData_matrix[batch_index]
            if sparse.isspmatrix(T):
                T = T.toarray()

            D = A + T
            device = torch.device('cuda')
            model.train().to(device)
            loss, _, = model(T, A, D)
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            update_count += 1

        if epoch % 2 == 0:
            precision, recall, f1, ndcg, one_call, mrr = [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], \
                [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]
            for batch_num, sta_idx in enumerate(range(0, len(userList_train), args.batch_size)):
                end_idx = min(sta_idx + args.batch_size, len(userList_train))
                batch_index = userList_train[sta_idx: end_idx]

                if args.total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / args.total_anneal_steps)
                else:
                    anneal = anneal_cap

                A = auxiliaryData_matrix[batch_index]
                if sparse.isspmatrix(A):
                    A = A.toarray()

                T = targetData_matrix[batch_index]
                if sparse.isspmatrix(T):
                    T = T.toarray()

                D = A + T

                model.eval()
                testDict_batch = []
                for i in batch_index:
                    testDict_batch.append(testDict[i])

                if args.total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / args.total_anneal_steps)
                else:
                    anneal = anneal_cap
                model.anneal = anneal
                model.data_collect(T, A, D)
                model.process()
                batch_pred = model.xu_pred

                batch_pred[T.nonzero()] = -np.inf
                top_list = torch.topk(input=batch_pred, k=5)
                top_indices = top_list.indices
                (precision_batch, recall_batch, f1_batch, ndcg_batch, one_call_batch,
                 mrr_batch) = mytest.computeTopNAccuracy(testDict_batch, top_indices, topN)

                precision[0] += precision_batch[0] / len(userList_test)
                recall[0] += recall_batch[0] / len(userList_test)
                f1[0] += f1_batch[0] / len(userList_test)
                ndcg[0] += ndcg_batch[0] / len(userList_test)
                one_call[0] += one_call_batch[0] / len(userList_test)
                mrr[0] += mrr_batch[0] / len(userList_test)

            print('Epoch: %d Loss: %.4f' % (epoch, loss))
            print('Epoch: %d precision@5: %.4f recall@5: %.4f f1@5: %.4f ndcg@5: %.4f one_call@5: %.4f' % (
                epoch, precision[0], recall[0], f1[0], ndcg[0], one_call[0]))

            logger.info('Epoch: %d Loss: %.4f' % (epoch, loss))
            logger.info('Epoch: %d precision@5: %.4f recall@5: %.4f f1@5: %.4f ndcg@5: %.4f one_call@5: %.4f' % (
                epoch, precision[0], recall[0], f1[0], ndcg[0], one_call[0]))
            precision_list.append(precision[0])
            recall_list.append(recall[0])
            ndcg_list.append(ndcg[0])

            if best_ndcg5 < ndcg[0]:
                best_prec5, best_rec5, best_f15, best_ndcg5, best_1call5, best_iter = precision[0], recall[0], f1[0], \
                    ndcg[0], one_call[0], epoch
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= args.early_stop:
                    break
        print("End. Best Iteration %d: NDCG@5 = %.4f " % (best_iter, best_ndcg5))
        print("Precision@5: %.4f " % best_prec5)
        print("Recall@5: %.4f " % best_rec5)
        print("F1@5: %.4f " % best_f15)
        print("NDCG@5: %.4f " % best_ndcg5)
        print("1-call@5: %.4f " % best_1call5)

        logger.info("End. Best Iteration %d: NDCG@5 = %.4f " % (best_iter, best_ndcg5))
        logger.info("Precision@5: %.4f " % best_prec5)
        logger.info("Recall@5: %.4f " % best_rec5)
        logger.info("F1@5: %.4f " % best_f15)
        logger.info("NDCG@5: %.4f " % best_ndcg5)
        logger.info("1-call@5: %.4f " % best_1call5)

    return precision_list, recall_list, ndcg_list, len(precision_list)


if __name__ == '__main__':
    args = parser_args()
    setup_seed(20)
    targetData_matrix, targetDict, usersNum, itemsNum = load_targetdata(args=args)
    auxiliaryData_matrix, auxiliaryDict = load_auxiliary_data(args=args)
    testDict = loadTestData(args=args)
    userList_train = sorted(list(set(targetDict.keys()).union(set(auxiliaryDict.keys()))))
    userList_test = sorted(testDict.keys())
    args.user_num, args.item_num = usersNum, itemsNum
    criterion = torch.nn.CrossEntropyLoss()
    p_dims = []
    if type(args.hiddenDim) == list:
        p_dims = args.hiddenDim
        p_dims.append(args.item_num)
    else:
        p_dims.append(args.hiddenDim)
        p_dims.append(args.item_num)

    model = VAE(item_num=p_dims[-1], hidden_size=p_dims[0], batch_size=args.batch_size, top_k=5,
                dropout_rate=0.5, lr=1e-3)
    model = model.to(model.device)
    # model = MultiVAE(p_dims=p_dims,dropout_p=0.5)
    # model.build_graph()
    # for name, parms in model.named_parameters():
    #     print('-->name: ,name')
    #     print('-->para:', parms)
    #     print('-->grad_requins:',parms.requires_grad)
    #     print('-->grad_value:', parms.grad)
    #     print("===")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)

    log_dir = './Log/' + args.dataset + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.%f')
    logger = logging.getLogger('Log')  # define a logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        filename=os.path.join(log_dir,
                              "VAEplusplus_%s_%s_%s_batch%d_hidden%s_reg%.4f_lr%.4f-%s.res" % (
                                  args.dataset, args.test, args.examination, args.batch_size, str(p_dims),
                                  args.reg_scale, args.lr_rate, timestamp)), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(args)  # if args normally loaded will print

    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    logger.info('Start time: %s' % timestamp)

    precision_list, recall_list, ndcg_list, length = train(args, criterion, model, optimizer,
                                                           targetData_matrix,  auxiliaryData_matrix, userList_train,
                                                           testDict, userList_test)
    data_visualization(recall_list, precision_list, ndcg_list, length)

    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    logger.info('End time: %s' % timestamp)
