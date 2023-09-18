"""
@author: bochengz
@date: 2023/04/15
@email: bochengzeng@bochengz.top
"""
from config import Config
from accelerate import Accelerator
from kogger import Logger
import pprint
from model import ToyModel
from dataset import ToyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time


def train(model, writer, accelerator, data_loader, epochs, loss_func, optimizer, scheduler, checkpoint_path, logger):

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_idx, (data, label) in enumerate(data_loader):
            # data: [b, n]
            output = model(data)
            loss = loss_func(output, label)
            total_loss = total_loss + loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
        scheduler.step()

        mean_loss = total_loss / len(data_loader)
        if accelerator.is_main_process:
            writer.add_scalar('Loss/train', mean_loss, epoch)

        if epoch == 1 or epoch % 10 == 0:
            accelerator.save_state(output_dir=checkpoint_path)
            if accelerator.is_main_process:
                logger.info('[Train {}/{}] MSE loss: {:.4e}'.format(epoch, epochs, mean_loss))


def main():
    # load and set config
    args = Config.get_parser().parse_args()
    config = Config(yaml_filename=args.filename)

    accelerator = Accelerator()

    # logger = Logger('PID %d' % accelerator.process_index, file=config['log_file'])
    logger = Logger('PID %d' % accelerator.process_index)
    if accelerator.is_main_process:
        logger.info('Load config successfully!')
        logger.info(pprint.pformat(config.data))

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

    writer = SummaryWriter(comment=config['experiment_name'])

    train(
        model=model,
        writer=writer,
        accelerator=accelerator,
        data_loader=data_loader,
        epochs=config['epochs'],
        loss_func=mse_func,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=config['checkpoint_path'],
        logger=logger
    )
    writer.flush()
    writer.close()

    if accelerator.is_main_process:
        # Test in single GPU
        datasetTest = ToyDataset(length=1000)
        test_loader = DataLoader(
            dataset=datasetTest,
            shuffle=config['shuffle'],
            batch_size=config['batch_size']
        )
        accelerator.load_state(input_dir=config['checkpoint_path'])
        test_losses = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                predictions = model(inputs)
                test_loss = mse_func(predictions, targets)
                test_losses.append(test_loss.item())

        logger.info('Test loss: {:.4e}'.format(sum(test_losses) / len(test_losses)))
        logger.info('Done')


if __name__ == '__main__':
    main()
