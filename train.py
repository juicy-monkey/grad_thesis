import torch
from torch import nn
from torch.utils.data import DataLoader


def _calc_loss(model, data_loader, device, optimizer):
    criterion = nn.BCELoss()

    for data in data_loader:
        inputs, targets = data[0].to(device), data[1].to(device)
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del inputs
        del targets

        return loss.item()


def train(net, data, num_epoch, device, batch_size=4, shuffle=True, threshold=1e-3, optimizer=None):
    model = net.to(device)
    model.train()
    loss_graph = []
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    for e in range(num_epoch):
        loss = _calc_loss(model, loader, device, optimizer)
        loss_graph.append(loss)
        print("Epoch: {} Loss: {}".format(e, loss))
        if loss <= threshold:
            break

    net.to('cpu')
    return loss_graph, e+1

