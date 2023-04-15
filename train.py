"""
@author: bochengz
@date: 2023/04/15
@email: bochengzeng@bochengz.top
"""
import config as cfg
from accelerate import Accelerator
from accelerate.logging import get_logger
from kogger import Logger
import pprint
from model import ToyModel
from dataset import ToyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np


def train(model, accelerator, data_loader, epochs, loss_func, optimizer, scheduler, checkpoint_path, logger):
    train_losses = []

    for ii in range(1, epochs+1):
        for batch_idx, (data, label) in enumerate(data_loader):
            # data: [b, n]
            output = model(data)
            loss = loss_func(output, label)

            # Backward and optimize
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if ii == 1 or ii % 10 == 0:
            accelerator.save_state(output_dir=checkpoint_path)
            if accelerator.is_main_process:
                logger.info('[Train {}/{}] MSE loss: {:.4e}'.format(ii, epochs, loss))

    return train_losses


if __name__ == '__main__':
    # load and set config
    args = cfg.get_parser().parse_args()
    config = cfg.load_config(yaml_filename=args.filename)
    config = cfg.process_config(config)

    accelerator = Accelerator()

    logger = Logger('PID %d' % accelerator.process_index, file=config['log_file'])
    if accelerator.is_main_process:
        logger.info('Load config successfully!')
        logger.info(pprint.pformat(config))

    dataset = ToyDataset(length=1000)

    data_loader = DataLoader(
        dataset=dataset,
        shuffle=config['shuffle'],
        batch_size=config['batch_size']
    )

    model = ToyModel()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    mse_func = torch.nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['steplr_size'], gamma=config['steplr_gamma'])
    model, optimizer, data_loader, scheduler = accelerator.prepare(
       model, optimizer, data_loader, scheduler
    )

    train_loss = train(
        model=model,
        accelerator=accelerator,
        data_loader=data_loader,
        epochs=config['epochs'],
        loss_func=mse_func,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=config['checkpoint_path'],
        logger=logger
    )

    # plot train loss
    fig = plt.figure()
    train_idx = np.arange(1, config['epochs']+1)
    # plt.plot(train_idx[int(len(train_idx)*0.2):-1], train_loss[int(len(train_idx)*0.2):-1], label='Train')  # 只绘制后0.8
    plt.semilogy(train_idx, train_loss)
    plt.title('Train Loss')
    fig.savefig(config['figs_loss_train'])

    # Test in single GPU
    datasetTest = ToyDataset(length=1000)
    test_loader = DataLoader(
        dataset=datasetTest,
        shuffle=config['shuffle'],
        batch_size=config['batch_size']
    )
    accelerator.load_state(input_dir=config['checkpoint_path'])
    test_losses = []
    for inputs, targets in test_loader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        predictions = model(inputs)
        test_loss = mse_func(predictions, targets)
        test_losses.append(test_loss.item())

    if accelerator.is_main_process:
        logger.info('Test loss: {:.4e}'.format(sum(test_losses) / len(test_losses)))

    logger.info('Done!')