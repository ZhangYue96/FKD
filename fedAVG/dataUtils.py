import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class ClientDataSet(Dataset):
    def __init__(self, dataset, idxs):
        # 定义好 image 的路径
        self.dataset = dataset
        self.idxs = list(idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label

    def __len__(self):
        return len(self.idxs)


def mnist_data(num_users, isIID):
    mnist_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=mnist_trans)
    if isIID:
        num_items = 600
        dict_users, all_idxs = {}, [i for i in range(len(trainset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
    else:
        num_shards, num_imgs = 200, 300
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        # idxs = np.arange(num_shards * num_imgs)
        labels = trainset.targets.numpy()
        print(labels.shape)
        idxs = np.argsort(labels)
        print(idxs[:20])

        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users, trainset


def cifar_data(num_users):
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=trans_cifar)
    num_items = 500
    dict_users, all_idxs = {}, [i for i in range(len(trainset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users, trainset



def get_dataLoader(user, num_users, isIID, dataset, batch_size):
    if dataset == 'mnist':
        dict_users, trainset = mnist_data(num_users, isIID)
    else:
        dict_users, trainset = cifar_data(num_users)

    clientDataset = ClientDataSet(trainset, dict_users[user])
    train_loader = DataLoader(clientDataset, batch_size=batch_size, shuffle=True)
    return train_loader


# loader = get_dataLoader(1, 10, True, 'cifar', 50)
# for i, (x, y) in enumerate(loader):
#     print(x.shape)
#     print(i)