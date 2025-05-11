import torch
import torch.optim as optim
import numpy as np
from .Net.CEAH_2 import CEAH
from .DataLoader.Data_Loader import MyDataset
import scipy.io as sio
from torch.utils.data import DataLoader
import sys
from pytorch_metric_learning import losses

sys.stdout = open('out.txt', 'w')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 80
learning_rate = 0.001
batch_size = 32
patch_size = 25
patch_num = 60

img_path = "./final_ADNI/"
data = sio.loadmat('data.mat')
sample_name = data['samples_train'].flatten()
labels = data['labels_train'].flatten()
# load patch location proposals calculated on training samples
template_cors = sio.loadmat('final_cors.mat')
template_cors = template_cors['patch_centers']

# build model
model = CEAH(patch_num=patch_num)
model.load_state_dict(torch.load("best_model.pkl", map_location='cpu'), strict=False)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

a = []
best = 9999
# train
for e in range(epoch):
    print('-----------------------------------------------------')
    print('epoch{}'.format(e))
    dataset = MyDataset(img_path, sample_name, labels, template_cors, patch_size, patch_num, shuffle_flag=True)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    model.train()
    l = []

    for batch_idx, (inputs, outputs, name) in enumerate(train_loader):
        inputs = inputs.to(device)
        inputs = inputs.permute(1, 0, 2, 3, 4, 5)
        subject_outputs = outputs.long().flatten().to(device)

        model.zero_grad()
        optimizer.zero_grad()

        emb, fff = model(inputs)
        cont_loss_func = losses.NTXentLoss(0.05).to(device)
        loss = cont_loss_func(emb, subject_outputs)

        loss.backward()
        optimizer.step()

        l.append(loss.item())

    print("总损失")
    print(np.mean(l))

    a.append(np.mean(l))
    if(np.mean(l) < best):
        best = np.mean(l)
        torch.save(model.state_dict(), 'model/best_model.pkl')
    print(a)
