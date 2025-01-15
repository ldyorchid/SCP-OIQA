import os, argparse, time

import numpy as np
import pandas as pd
import random
import torch


import torch.backends.cudnn as cudnn

import dataset_5
import models_1
from torchvision import transforms

from scipy import stats
from scipy.optimize import curve_fit
from scipy.io import savemat

from torch.autograd import Variable

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]

    popt, _ = curve_fit(logistic_func,y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="No reference 360 degree image quality assessment.")
    parser.add_argument('--gpu', dest='gpu_id', help="GPU device id to use [0]", default=0, type=int)  #首先是一个name，dest是用来指定参数的位置，default是参数未赋值时的默认值，help是用来描述
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=10, type=int)

    parser.add_argument('--database', dest='database', help='The database that needs to be trained and tested.',
          default='CVIQ', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=10, type=int)
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')

    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')

    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='E:\MC360IQA-main\OIQA_database\OIQA', type=str)
    parser.add_argument('--filename_train', dest='filename_train', help='Training csv file containing relative paths for every example.',
          default=r"E:\SCP-OIQA\OIQA\OIQA_train_8.csv", type=str)
    parser.add_argument('--filename_test', dest='filename_test', help='Test csv file containing relative paths for every example.',
          default=r"E:\SCP-OIQA\OIQA\OIQA_test_8.csv", type=str)
    parser.add_argument('--cross_validation_index', dest='cross_validation_index', help='The index of cross validation.',
          default='0', type=int)    #此处的参数作用是啥？？
    # 模型路径
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)    #此处的作用是啥？？？？

    args = parser.parse_args()

    return args




def train(args):
    cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    snapshot = args.snapshot
    database = args.database
    filename_train = args.filename_train
    filename_test = args.filename_test

    cross_validation_index = args.cross_validation_index
    torch.cuda.set_device(gpu)


    if not os.path.exists(os.path.join(snapshot, database, str(cross_validation_index))):
        os.makedirs(os.path.join(snapshot, database, str(cross_validation_index)))


    transformations = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), \
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    train_dataset = dataset_5.Dataset(args.data_dir, filename_train, transformations)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    # 加载测试数据
    test_dataset = dataset_5.Dataset(args.data_dir, filename_test, transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                              num_workers=0)
    model_hyper = models_1.HyperNet(16, 768, 224, 112,\
                  56, 28, 14, 7).cuda()


    #model_hyper.load_state_dict((torch.load(r'E:\SCP-OIQA\OIQA\final\OIQA_epoch_1.pkl')))
    l1_loss = torch.nn.MSELoss().cuda()  # MAE



    backbone_params = list(map(id, model_hyper.res.parameters()))


    hypernet_params = filter(lambda p: id(p) not in backbone_params, model_hyper.parameters())

    lr = args.lr  # 2e-5
    lrratio = args.lr_ratio  # 10

    paras = [{'params': hypernet_params, 'lr': lr * lrratio},
             {'params': model_hyper.res.parameters(), 'lr': lr },
             ]
    optimizer = torch.optim.RMSprop(paras,  alpha=0.9)

    """Training"""
    best_srcc = 0.0
    best_plcc = 0.0


    for t in range(num_epochs):
        epoch_loss = []
        pred_scores = []
        gt_scores = []
        model_hyper.train(True)

        for batch_idx, batch in enumerate(train_loader):
            data1, data2, label = batch
            data1 = Variable(data1.cuda())
            data2 = Variable(data2.cuda())
            label = Variable(label.cuda())

            optimizer.zero_grad()
            # Generate weights for target network
            paras = model_hyper(data1, data2)  # 'paras' contains the network weights conveyed to target network

            #Building target network
            model_target = models_1.TargetNet(paras).cuda()

            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
            pred = pred.view(min(batch_size, len(pred)//6), -1).mean(axis=1)

            pred_scores = pred_scores + pred.cpu().tolist()
            gt_scores = gt_scores + label.cpu().squeeze().tolist()
            loss = l1_loss(pred, label.squeeze())
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)



        with torch.no_grad():
            model_hyper.eval()

            pred_scores = []
            gt_scores = []

            for batch_idx, batch in enumerate(test_loader):
                # Data.
                data1, data2, label = batch
                data1 = data1.cuda()
                data2 = data2.cuda()
                label = label.cuda()

                paras = model_hyper(data1, data2)
                model_target = models_1.TargetNet(paras).cuda()

                model_target.train(False)

                pred = model_target(paras['target_in_vec'])


                pred = pred.mean()

                pred_scores.append(pred.item())
                gt_scores.append(label.item())

            gt_scores = np.array(gt_scores)
            label = gt_scores.reshape(int(len(test_dataset) / 180), 180)  # 180去掉了
            label = np.mean(label, axis=1)

            pred_scores = np.array(pred_scores)
            y_put = pred_scores.reshape(int(len(test_dataset) / 180), 180)  # 180去掉了
            y_put = np.mean(y_put, axis=1)



            y_output_logistic = fit_function(label, y_put)

            test_srcc, _ = stats.spearmanr(y_put, label)
            test_plcc, _ = stats.pearsonr(y_output_logistic, label)
            test_RMSE = np.sqrt(((y_output_logistic - label) ** 2).mean())

        if test_srcc + test_plcc > best_srcc + best_plcc:
            best_srcc = test_srcc
            best_plcc = test_plcc
        # if test_srcc > 0.96:
        #     savemat(os.path.join(snapshot, database, str(cross_validation_index),database + '_epoch_' + str(t + 1) + '.mat'), {'label': label, 'score': y_put})
        #     # torch.save(model_hyper.state_dict(), os.path.join(snapshot, database, str(cross_validation_index),
        #     #                                             database + '_epoch_' + str(t + 1) + '.pkl'))
        print('epoch: %d\tloss: %4.3f\t\ttrain_srcc: %4.4f\t\ttest_srcc: %4.4f\t\ttest_plcc: %4.4f\t\ttest_RMSE: %4.4f' %
              (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_RMSE))


        # Update optimizer
        #lr = lr / pow(10, (t // 4))
        #if t > 1:
        # lr = lr / 10
        # lrratio = 10
        # paras = [{'params': hypernet_params, 'lr': lr * lrratio},
        #               {'params': model_hyper.res.parameters(), 'lr': lr}
        #               ]

        # optimizer = torch.optim.RMSprop(paras, alpha=0.9)
if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    args = parse_args()
    for i in range(100):
        train(args)










