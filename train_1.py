import torch
import torch.optim as optim
import numpy as np
from Net.CEAH import CEAH
from DataLoader.Data_Loader import MyDataset
import scipy.io as sio
from torch.utils.data import DataLoader
import sys
import random
from tqdm import tqdm
def main():
    sys.stdout = open('out_pMCIsMCI_1e1.txt', 'w')
    # def set_seed(seed):
    #     torch.manual_seed(seed)=
    #     torch.cuda.manual_seed_all(seed)
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     torch.backends.cudnn.deterministic = True

    # set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epoch = 50
    batch_size = 8

    learning_rate = 1
    patch_size = 25
    patch_num = 60

    img_path = "./final_ADNI/"

    data = sio.loadmat('data_3.mat')
    sample_name = data['samples_train'].flatten()
    labels = data['labels_train'].flatten()

    # load patch location proposals calculated on training samples
    template_cors = sio.loadmat('final_cors.mat')
    template_cors = template_cors['patch_centers']

    # build model
    model = CEAH(patch_num=patch_num)
    model = model.to(device)
    print("The total number of parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best = 0
    wrong_sam = []
    a_1 = []
    a_2 = []

    # train
    for e in range(epoch):
        print('-----------------------------------------------------')
        print('epoch{}'.format(e))
        dataset = MyDataset(img_path, sample_name, labels, template_cors, patch_size, patch_num, shuffle_flag=True)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        model.train()
        l = []
        a = []
        for batch_idx, (inputs, outputs, name) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)
            inputs = inputs.permute(1, 0, 2, 3, 4, 5)
            subject_outputs = outputs.long().flatten().to(device)

            model.zero_grad()
            optimizer.zero_grad()
            subject_pred = model(inputs)

            loss = criterion(subject_pred, subject_outputs)

            loss.backward()
            optimizer.step()

            l.append(loss.item())
            subject_pred = subject_pred.max(1)[1]

            acc = torch.sum(torch.eq(subject_pred, subject_outputs)).cpu().numpy()
            a.append(acc / batch_size)

        print("Loss:")
        print(np.mean(l))
        print("Training ACC")
        print(np.mean(a))

        a_1.append(np.mean(l))
        a_2.append(np.mean(a))

        torch.save(model.state_dict(), 'model/best_model_epoch{}.pkl'.format(e))
        print(a_1)
        print(a_2)

if __name__ == '__main__':
    main()