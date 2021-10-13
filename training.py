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
from word_tokens import token_EOS

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import csv
from inference import testModel
from pathlib import Path
from math import cos, pi

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

def compute_model_accuracy(model, dataset, encoding_file, device, numbers_accuracy_epsilon = 0.01):
    data = DataLoader(dataset, 1, True)

    script_true = 0
    script_false = 0
    numbers_true = 0
    numbers_false = 0

    model.eval()
    for iteration, (sample_batched) in enumerate(data):
        image = sample_batched['render'].to(device)
        # get targets (as python lists) to compare with the output
        targets_script = sample_batched['target_corpus'].to(device)[0]
        targets_numbers = sample_batched['target_numbers'].to(device)[0]
        
        with torch.no_grad():
            s_t, s_f, n_t, n_f = model.compute_accuracy(image, targets_script, targets_numbers, numbers_accuracy_epsilon)

        script_true += s_t
        script_false += s_f
        numbers_true += n_t
        numbers_false += n_f

        print("{:.2f}%".format(iteration / len(dataset) * 100))

    return script_true / (script_true + script_false), numbers_true / (numbers_true + numbers_false)

def validate_model(model, dataset, device, global_step, writer, out_val_script_path, out_val_numbers_path):
    model.eval()
    validation_loss_script = 0.0
    validation_loss_numbers = 0.0

    loss_fn_corpus = nn.CrossEntropyLoss()
    loss_fn_numbers = nn.MSELoss()

    for iteration, sample_batched in enumerate(dataset):
        sources = sample_batched['render'].to(device)
        targets_script = sample_batched['target_corpus'].to(device)
        targets_numbers = sample_batched['target_numbers'].to(device)

        enc_img = model.forward_Image_Encoder(sources)
        predicted_script = model.forward_Corpus_Decoder(targets_script[:, :-1], enc_img, target_mask=model.generate_square_subsequent_mask(len(targets_script[:, :-1])).to(device))
        if model.numbers_mlp:
            predicted_numbers = model.forward_Numbers_Decoder(None, enc_img, targets_script)
        else:
            predicted_numbers = model.forward_Numbers_Decoder(targets_numbers[:, :-1], enc_img, targets_script[:, :-1], model.generate_square_subsequent_mask(len(targets_numbers[:, :-1])).to(device))
        
        output_dim_script = predicted_script.shape[-1]
        output_dim_numbers = predicted_numbers.shape[-1]
        
        pred_script = predicted_script.contiguous().view(-1, output_dim_script)
        trg_script = targets_script[:, 1:].contiguous().view(-1)

        if model.numbers_mlp:
            pred_numbers = predicted_numbers.contiguous().view(-1)
            trg_numbers = targets_numbers.contiguous().view(-1)
            trg_numbers = torch.cat([trg_numbers, torch.zeros((pred_numbers.shape[0] - trg_numbers.shape[0])).to(device)], dim=0)

        else:
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

    print("Loss (script): {:.2f}".format(validation_loss_script))
    print("Loss (numbers): {:.2f}".format(validation_loss_numbers))

    write_scalar(out_val_script_path, writer, global_step, validation_loss_script)
    write_scalar(out_val_numbers_path, writer, global_step, validation_loss_numbers)

def evaluate(model, dataset, device, encoding_file, output_folder):
    data = DataLoader(dataset, 1, True)
    # create folder if not exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    model.eval()
    return testModel(model, data, device, encoding_file, output_folder)

def cos_annealing(iteration):
    annealing_rate = 250
    max_lr = 3e-4
    min_lr = 1e-6
    upper_lr = max_lr - min_lr
    return upper_lr / 2.0 * (cos(1.0 / annealing_rate * pi * (iteration % annealing_rate)) + 1) + min_lr

if __name__ == "__main__":
    #csv_path = "TestData/dataset.csv"
    #dataset_train_path = "/home/baldur/Dataset/ShapeNet/Dataset_Polygen_normalized/Train"
    #dataset_test_path = "/home/baldur/Dataset/ShapeNet/Dataset_Polygen_normalized/Test"
    #dataset_eval_path = "/home/baldur/Dataset/ShapeNet/Dataset_Polygen_normalized/Test"
    dataset_train_path = "/root/Datasets/House/Train"
    dataset_test_path = "/root/Datasets/House/Test"
    dataset_eval_path = "/root/Datasets/House/Test"

    csv_path_train = os.path.join(dataset_train_path, "dataset.csv")
    csv_path_test = os.path.join(dataset_test_path, "dataset.csv")
    csv_path_eval = os.path.join(dataset_test_path, "dataset_small.csv")
    file_ending_script = '.py'

    #model_folder = os.path.join(dataset_train_path, "models")
    model_folder = "models"
    encoding_file = "encoding.txt"
    encoding, max_len_encoding, max_len_floats = None, None, None
    epochs = 10
    batch_size = 4
    cosine_annealing = False
    temporary_freeze_corpus_decoder = True #TODO undo this when finished with this test
    writer = SummaryWriter(log_dir="graphs")

    out_training_script_path = "train_script.csv"
    out_training_numbers_path = "train_nbrs.csv"
    out_val_script_path = "val_script.csv"
    out_val_numbers_path = "val_nbrs.csv"
    out_acc_script_path = "acc_script.csv"
    out_acc_nbrs_path = "acc_nbrs.csv"
    
    with open(out_training_script_path, 'w+') as train_script:
        train_script.write("Step,Loss\n")
    with open(out_training_numbers_path, 'w+') as train_nbrs:
        train_nbrs.write("Step,Loss\n")
    with open(out_val_script_path, 'w+') as val_script:
        val_script.write("Step,Loss\n")
    with open(out_val_numbers_path, 'w+') as val_nbrs:
        val_nbrs.write("Step,Loss\n")
    with open(out_acc_script_path, 'w+') as acc_scr:
        acc_scr.write("Step,Loss\n")
    with open(out_acc_nbrs_path, 'w+') as acc_nbrs:
        acc_nbrs.write("Step,Loss\n")

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #encoding, max_len_encoding, max_len_floats = st.load_sentence_encoding(encoding_file)

    print("Creating dataset for training...")
    dataset_train = Dataset_ScriptGen(csv_path_train, dataset_train_path, encoding, max_len_encoding, max_len_floats, transforms.Compose([
            Rescale(256, 256),
            ToTensor()
    ]), file_ending_script)

    encoding, max_len_encoding, max_len_floats = dataset_train.encoding, dataset_train.max_len_encoding, dataset_train.max_len_floats

    # save encoding from training -> use it for evaluation later
    st.save_sentence_encoding(dataset_train.encoding, dataset_train.max_len_encoding, dataset_train.max_len_floats, encoding_file)

    print("Creating dataset for validation...")
    dataset_test = Dataset_ScriptGen(csv_path_test, dataset_test_path, encoding, max_len_encoding, max_len_floats, transforms.Compose([
            Rescale(256, 256),
            ToTensor()
    ]), file_ending_script)

    print("Creating dataset for evaluation...")
    dataset_eval = Dataset_ScriptGen(csv_path_eval, dataset_eval_path, encoding, max_len_encoding, max_len_floats, transforms.Compose([
            Rescale(256, 256),
            ToTensor()
    ]), file_ending_script)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # for testing on cpu
    device = 'cpu'
    
    print("Using device:", device)

    train_data = DataLoader(dataset_train, batch_size, True)
    test_data = DataLoader(dataset_test, batch_size, True)

    model = SkriptGen(
        encoding = dataset_train.encoding, 
        max_len_encoding = dataset_train.max_len_encoding, 
        max_len_floats = dataset_train.max_len_floats,
        hidden_size = 256,
        pretrained_resnet = False,
        fixed_pe = False,
        numbers_mlp = False,
        device = device,
        )
    print("amount of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    model.init_weights()

    #'''
    loss_fn_corpus = nn.CrossEntropyLoss()
    loss_fn_numbers = nn.MSELoss()
    '''
    loss_fn_corpus = nn.CrossEntropyLoss(reduction="sum")
    loss_fn_numbers = nn.MSELoss(reduction="sum")
    #'''

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

            if cosine_annealing:
                for g in optimizer.param_groups:
                    g['lr'] = cos_annealing(iteration)

            enc_img = model.forward_Image_Encoder(sources)
            predicted_script = model.forward_Corpus_Decoder(targets_script[:, :-1], enc_img, target_mask=model.generate_square_subsequent_mask(len(targets_script[:, :-1])).to(device))
            if model.numbers_mlp:
                predicted_numbers = model.forward_Numbers_Decoder(None, enc_img, targets_script)
            else:
                predicted_numbers = model.forward_Numbers_Decoder(targets_numbers[:, :-1], enc_img, targets_script[:, :-1], model.generate_square_subsequent_mask(len(targets_numbers[:, :-1])).to(device))

            output_dim_script = predicted_script.shape[-1]
            output_dim_numbers = predicted_numbers.shape[-1]
            
            pred_script = predicted_script.contiguous().view(-1, output_dim_script)
            trg_script = targets_script[:, 1:].contiguous().view(-1)

            if model.numbers_mlp:
                pred_numbers = predicted_numbers.contiguous().view(-1)
                trg_numbers = targets_numbers.contiguous().view(-1)
                trg_numbers = torch.cat([trg_numbers, torch.zeros((pred_numbers.shape[0] - trg_numbers.shape[0])).to(device)], dim=0)

            else:
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
            if e < 1: #write gradients in first epoch
                #print("writing gradients...")
                write_grad_flow(model.named_parameters(),
                    "/root/ScriptGen/gradients/grad_flow_ep_{}_it_{}.csv".format(e, iteration),
                    #"./gradients/grad_flow_ep_{}_it_{}.csv".format(e, iteration),
                    "Gradient flow at iteration {}".format(iteration))

            optimizer.step()
            global_step += 1

            # for testing
            #if iteration > 1:
            #    break
            
        print('Validating...')
        validate_model(model, test_data, device, global_step, writer, out_val_script_path, out_val_numbers_path)
        print('Computing accuracy...')
        mean_acc_script, mean_acc_nbrs = compute_model_accuracy(model, dataset_test, encoding_file, device)
        write_scalar(out_acc_script_path, writer, global_step, mean_acc_script)
        write_scalar(out_acc_nbrs_path, writer, global_step, mean_acc_nbrs)
        save_checkpoint(model, "SkriptGen-Ep" + str(e), model_folder, 0, True)

        print('Evaluating...')
        evaluate(model, dataset_eval, device, encoding_file, './evaluate/epoch-{}'.format(e))