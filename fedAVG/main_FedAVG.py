import torch
import numpy as np
from net import *
from dataUtils import *
from FedAVG_test import *
import copy
import matplotlib.pyplot as plt


# dataset args
dataset = 'mnist'
isIID = True
# model agruments
model = 'cnn'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# federated args
num_users = 100
fraction = 0.1
epochs = 10
local_epo = 5
local_batch = 10
lr = 0.01
momentum = 0.5



def clientUpdate(user, net, global_params):
    net.load_state_dict(global_params)
    clientDataset = ClientDataSet(trainset, dict_users[user])
    train_loader = DataLoader(clientDataset, batch_size=local_batch, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(local_epo):
        epoch_loss = []
        for batch_id, (data, label) in enumerate(train_loader):
            batch_loss = []
            data, label = data.to(device), label.to(device)
            net.zero_grad()
            preds = net(data)
            loss = loss_func(preds, label)
            loss.backward()
            optimizer.step()
            if batch_id % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_id * len(data), len(train_loader.dataset),
                          100. * batch_id / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    local_loss = sum(epoch_loss) / len(epoch_loss)
    local_params = net.state_dict()
    # print(type(local_params))
    # print(len(local_params))
    # print(local_params)
    return local_params, local_loss

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    # 参数格式为dict，key为每层网络名称，weight和bias分两个key
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


if __name__ == '__main__':
    # build dataset
    if dataset == 'mnist':
        dict_users, trainset = mnist_data(num_users, isIID)
    else:
        dict_users, trainset = cifar_data(num_users)
    # build model
    if model == 'cnn' and dataset == 'cifar10':
        net_glob = CNNCifar().to(device)
    elif model == 'cnn' and dataset == 'mnist':
        net_glob = CNNMnist().to(device)
    elif model == 'mlp':
        if dataset == 'mnist':
            input_size = 28*28
        else:
            input_size = 3*32*32
        net_glob = MLP(dim_in=input_size, dim_hidden=200).to(device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # FedAVG train
    net_glob.train()
    global_params = net_glob.state_dict()
    loss_train = []

    for iter in range(epochs):
        m = max(int(fraction*num_users), 1)
        user_idxs = np.random.choice(range(num_users), m, replace=False)
        local_loss = []
        local_w = []
        for idx in user_idxs:
            local_param, loss = clientUpdate(idx, net_glob, global_params)
            local_w.append(copy.deepcopy(local_param))
            local_loss.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(local_w)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(local_loss) / len(local_loss)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.show()

    # testing
    mnist_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=mnist_trans)
    testset = datasets.MNIST('../data/mnist/', train=False, download=True, transform=mnist_trans)

    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, trainset, device, 64)
    acc_test, loss_test = test_img(net_glob, testset, device, 64)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))






