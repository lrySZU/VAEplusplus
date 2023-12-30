import numpy as np
import pandas as pd
from scipy import sparse


def load_targetdata(args):
    file = args.path + '/' + args.dataset + '/' + args.transaction
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    usersNum, itemsNum = args.user_num + 1, args.item_num + 1

    """
    groupby('uid')['iid']表示按照uid列进行分组，并选择iid列的数据。
    apply(list)将每个分组中的iid列的数据转换为列表形式。
    最后，to_dict()将分组结果转换为字典格式，其中字典的键是uid值，而字典的值是对应的iid列表
    """
    targetDict = tp.groupby('uid')['iid'].apply(list).to_dict()  # hashset

    rows, cols = tp['uid'], tp['iid']
    """
    (np.ones_like(rows), (rows, cols)): 这个参数传递了非零元素的值和它们在矩阵中的位置。
    np.ones_like(rows)表示将所有非零元素的值设为1，
    (rows, cols)表示非零元素在矩阵中的行和列索引
    """
    targetData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, itemsNum)
    )  # matrix
    return targetData, targetDict, usersNum, itemsNum


def load_auxiliary_data(args):
    file = args.path + '/' + args.dataset + '/' + args.examination
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values('uid')
    usersNum, itemsNum = args.user_num + 1, args.item_num + 1
    auxiliaryDict = tp.groupby('uid')['iid'].apply(list).to_dict()

    rows, cols = tp['uid'], tp['iid']
    auxiliaryData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, itemsNum))
    return auxiliaryData, auxiliaryDict


def loadTestData(args):
    file = args.path + '/' + args.dataset + '/' + args.test
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    testDict = tp.groupby('uid')['iid'].apply(list).to_dict()
    return testDict
