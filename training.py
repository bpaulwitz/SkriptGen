import torch
from models import SkriptGen
#from models2 import SkriptGen
from dataset import Dataset_ScriptGen, Rescale, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import sentence_tokens as st
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import csv
from inference import testModel
from pathlib import Path

#see https://github.com/tunz/transformer-pytorch/blob/e7266679f0b32fd99135ea617213f986ceede056/utils/utils.py#L34
def save_checkpoint(model, name, filepath, global_step, is_best):
    model_save_path = filepath + '/' + name
    torch.save(model, model_save_path)
    torch.save(global_step, filepath + '/global_step.pt')
    if is_best:
        best_save_path = filepath + '/best_model.pt'
        shutil.copyfile(model_save_path, best_save_path)

#see https://github.com/tunz/transformer-pytorch/blob/e7266679f0b32fd99135ea617213f986ceede056/utils/utils.py#L43
def load_checkpoint(model_path, device, is_eval=True):
    if is_eval:
        model = torch.load(model_path + '/best_model.pt')
        model.eval()
        return model.to(device=device)

    model = torch.load(model_path + '/last_model.pt')
    global_step = torch.load(model_path + '/global_step.pt')
    return model.to(device=device), global_step

def write_scalar(csv_path: str, tensorboard_writer: SummaryWriter, global_step: int, value):
    with open(csv_path, 'a') as csv_file:
        csv_file.write("{},{}\n".format(global_step, value))
    tensorboard_writer.add_scalar(csv_path, value, global_step)

def write_grad_flow(named_parameters, filename_csv, title):
    ave_max_layer_rows = []
    i = 0
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None: print(i,n,p,p.grad)
            ave_max_layer_rows.append([p.grad.abs().mean().item(), p.grad.abs().max().item(), n.replace('.weight', '')])
        i += 1

    with open(filename_csv, 'w+') as csv_file:
        csv_file.write(title + '\n')
        csv_file.write("ave_grads,max_grads,layers\n")

        writer = csv.writer(csv_file)
        writer.writerows(ave_max_layer_rows)


# from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow_h(named_parameters, save_as = None, title = None, file_format = 'png'):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    #plt.figure(figsize=[48, 27], dpi=80) # 4k
    #plt.figure(figsize=[24, 13.5], dpi=80) # full hd
    plt.figure(figsize=[30, 30], dpi=80)
    ave_grads = []
    max_grads= []
    layers = []
    i = 0
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None: print(i,n,p,p.grad)
            layers.append(n.replace('.weight', ''))
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
        i += 1

    plt.barh(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.barh(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.vlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.yticks(range(0,len(ave_grads), 1), layers, rotation="horizontal")
    plt.ylim(bottom=0, top=len(ave_grads))
    plt.xlim(left = -0.001, right=0.02) # zoom in on the lower gradient regions
    plt.ylabel("Layers")
    plt.xlabel("average gradient")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    if save_as is not None:
        plt.savefig(save_as, format=file_format)
    plt.close()


def validate_model(model, dataset, device, global_step, writer, out_val_script_path, out_val_numbers_path):
    model.eval()
    validation_loss_script = 0.0
    validation_loss_numbers = 0.0

    for iteration, sample_batched in enumerate(dataset):
        sources = sample_batched['render'].to(device)
        targets_script = sample_batched['target_corpus'].to(device)
        targets_numbers = sample_batched['target_numbers'].to(device)

        predicted_script, predicted_numbers = model(sources, targets_script[:, :-1], targets_numbers[:, :-1])

        output_dim_script = predicted_script.shape[-1]
        output_dim_numbers = predicted_numbers.shape[-1]
        
        pred_script = predicted_script.contiguous().view(-1, output_dim_script)
        trg_script = targets_script[:, 1:].contiguous().view(-1)

        pred_numbers = predicted_numbers.contiguous().view(-1, output_dim_numbers)[:,0]
        trg_numbers = targets_numbers[:, 1:].contiguous().view(-1)

        loss_script = loss_fn_corpus(pred_script, trg_script)
        loss_numbers = loss_fn_numbers(pred_numbers, trg_numbers)

        validation_loss_script += loss_script.item()
        validation_loss_numbers += loss_numbers.item()
        print("{:.2f}%".format(iteration / len(dataset) * 100))

        # for testing
        #if iteration > 2:
        #        break


    validation_loss_script /= len(dataset)
    validation_loss_numbers /= len(dataset)

    write_scalar(out_val_script_path, writer, global_step, validation_loss_script)
    write_scalar(out_val_numbers_path, writer, global_step, validation_loss_numbers)
    print(validation_loss_script, validation_loss_numbers)

def evaluate(model, dataset, device, encoding_file, output_folder):
    data = DataLoader(dataset, 1, True)
    # create folder if not exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    model.eval()
    testModel(model, data, device, encoding_file, output_folder)

if __name__ == "__main__":
    #csv_path = "TestData/dataset.csv"
    #dataset_train_path = "/home/baldur/Dataset/ShapeNet/HouseDataset2_Polygen/Train"
    #dataset_test_path = "/home/baldur/Dataset/ShapeNet/HouseDataset2_Polygen/Test"
    dataset_train_path = "/root/Datasets/House/Train"
    dataset_test_path = "/root/Datasets/House/Test"
    dataset_eval_path = "/root/Datasets/House/Test"

    csv_path_train = os.path.join(dataset_train_path, "dataset.csv")
    csv_path_test = os.path.join(dataset_test_path, "dataset.csv")
    csv_path_eval = os.path.join(dataset_test_path, "dataset_small.csv")

    #model_folder = os.path.join(dataset_train_path, "models")
    model_folder = "models"
    encoding_file = "encoding.txt"
    encoding, max_len_encoding, max_len_floats = None, None, None
    epochs = 20
    batch_size = 4
    writer = SummaryWriter(log_dir="graphs")

    out_training_script_path = "train_script.csv"
    out_training_numbers_path = "train_nbrs.csv"
    out_val_script_path = "val_script.csv"
    out_val_numbers_path = "val_nbrs.csv"
    
    with open(out_training_script_path, 'w+') as train_script:
        train_script.write("Step,Loss\n")
    with open(out_training_numbers_path, 'w+') as train_nbrs:
        train_nbrs.write("Step,Loss\n")
    with open(out_val_script_path, 'w+') as val_script:
        val_script.write("Step,Loss\n")
    with open(out_val_numbers_path, 'w+') as val_nbrs:
        val_nbrs.write("Step,Loss\n")

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #encoding, max_len_encoding, max_len_floats = st.load_sentence_encoding(encoding_file)

    print("Creating dataset for training...")
    dataset_train = Dataset_ScriptGen(csv_path_train, dataset_train_path, encoding, max_len_encoding, max_len_floats, transforms.Compose([
            Rescale(256, 256),
            ToTensor()
    ]))

    encoding, max_len_encoding, max_len_floats = dataset_train.encoding, dataset_train.max_len_encoding, dataset_train.max_len_floats

    # save encoding from training -> use it for evaluation later
    st.save_sentence_encoding(dataset_train.encoding, dataset_train.max_len_encoding, dataset_train.max_len_floats, encoding_file)

    print("Creating dataset for validation...")
    dataset_test = Dataset_ScriptGen(csv_path_test, dataset_test_path, encoding, max_len_encoding, max_len_floats, transforms.Compose([
            Rescale(256, 256),
            ToTensor()
    ]))

    print("Creating dataset for evaluation...")
    dataset_eval = Dataset_ScriptGen(csv_path_eval, dataset_eval_path, encoding, max_len_encoding, max_len_floats, transforms.Compose([
            Rescale(256, 256),
            ToTensor()
    ]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # for testing on cpu
    #device = 'cpu'
    
    print("Using device:", device)

    train_data = DataLoader(dataset_train, batch_size, True)
    test_data = DataLoader(dataset_test, batch_size, True)

    model = SkriptGen(dataset_train.encoding, dataset_train.max_len_encoding, dataset_train.max_len_floats, 8, device)
    print("amount of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    model.init_weights()

    loss_fn_corpus = nn.CrossEntropyLoss()
    loss_fn_numbers = nn.MSELoss()

    #optimizer = Adam(params=model.parameters(), lr=0.00001, weight_decay=0.00008)
    optimizer = Adam(params=model.parameters(), lr=5e-5, weight_decay=0.00008)

    global_step = 0
    for e in range(epochs):
        model.train()
        for iteration, sample_batched in enumerate(train_data):
            sources = sample_batched['render'].to(device)
            targets_script = sample_batched['target_corpus'].to(device)
            targets_numbers = sample_batched['target_numbers'].to(device)
            optimizer.zero_grad()

            predicted_script, predicted_numbers = model(sources, targets_script[:, :-1], targets_numbers[:, :-1])

            output_dim_script = predicted_script.shape[-1]
            output_dim_numbers = predicted_numbers.shape[-1]
            
            pred_script = predicted_script.contiguous().view(-1, output_dim_script)
            trg_script = targets_script[:, 1:].contiguous().view(-1)

            pred_numbers = predicted_numbers.contiguous().view(-1, output_dim_numbers)[:,0]
            trg_numbers = targets_numbers[:, 1:].contiguous().view(-1)

            loss_script = loss_fn_corpus(pred_script, trg_script)
            loss_numbers = loss_fn_numbers(pred_numbers, trg_numbers)
            print("Epoch: {}\tIteration: {}\tL_Corpus: {:.5f}\tL_Numbers: {:.5f}".format(e, iteration, loss_script.item(), loss_numbers.item()))

            #lines.append("0," + str(iteration) + "," + str(loss_script.item()) + "," + str(loss_numbers.item()) + "\n")

            write_scalar(out_training_script_path, writer, global_step, loss_script.item())
            write_scalar(out_training_numbers_path, writer, global_step, loss_numbers.item())

            loss_script.backward(retain_graph=True)
            loss_numbers.backward()

            # Plotting / writing gradient flow
            #plot_grad_flow_h(model.named_parameters(), 
            #    "/home/baldur/Dataset/gradients/grad_flow_ep_{}_it_{}.svg".format(e, iteration), 
            #    "Gradient flow at iteration {}".format(iteration))
            if e <= 1: #write gradients in first two epochs
                #print("writing gradients...")
                write_grad_flow(model.named_parameters(),
                    "/root/ScriptGen/gradients/grad_flow_ep_{}_it_{}.csv".format(e, iteration),
                    "Gradient flow at iteration {}".format(iteration))

            optimizer.step()
            global_step += 1

            # for testing
            #if iteration > 1:
            #    break
            
        print('Validating...')
        validate_model(model, test_data, device, global_step, writer, out_val_script_path, out_val_numbers_path)
        save_checkpoint(model, "SkriptGen-Ep" + str(e), model_folder, 0, True)

        print('Evaluating...')
        evaluate(model, dataset_eval, device, encoding_file, './evaluate/epoch-{}'.format(e))
