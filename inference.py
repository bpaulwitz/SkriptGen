import os
import cv2
import torch
import numpy as np
import pandas as pd
import sentence_tokens as st
from dataset import Dataset_ScriptGen, Rescale, ToTensor, save_dataset_render
from torchvision import transforms
from torch.utils.data import DataLoader
from models import SkriptGen

def testModel(model, data, device, encoding_file, output_folder):
    encoder, max_length_encoding, max_length_numbers = st.load_sentence_encoding(encoding_file)
    decoder = st.decoding_from_encoding(encoder)
    for iteration, (sample_batched) in enumerate(data):
        image = sample_batched['render'].to(device)
        # get targets (as python lists) to compare with the output
        targets_script = sample_batched['target_corpus'].to(device)[0].tolist()
        targets_numbers = sample_batched['target_numbers'].to(device)[0].tolist()
        iterator = 0
        
        with torch.no_grad():
            enc_src = model.forward_Image_Encoder(image)

        output_indices_corpus = [int(encoder[st.token_SOS])]
        output_indices_numbers = [0]
        print("decoding script")
        for i in range(max_length_encoding):

            trg_tensor = torch.LongTensor(output_indices_corpus).unsqueeze(0).to(device)
            trg_mask = model.generate_square_subsequent_mask(len(trg_tensor)).to(device)

            with torch.no_grad():
                output = model.forward_Corpus_Decoder(trg_tensor, enc_src, trg_mask)

            prediction = output.argmax(2)[:,-1].item()
            iterator += 1
            #print(output.argmax(2)[:,:])
            if i < len(targets_script):
                ground_truth = targets_script[i]
            else:
                ground_truth = int(encoder[st.token_PAD])
            print("{:.2f}% -> Prediction:\t -> {}\t Groundtruth:\t -> {}".format(iterator / (max_length_encoding + max_length_numbers) * 100, prediction, ground_truth))

            output_indices_corpus.append(prediction)

            #if prediction == trg_field.vocab.stoi[trg_field.eos_token]:
                #break

        # create tensor from output_indices_corpus list
        corpus_output = torch.Tensor(output_indices_corpus)[None, ...].to(device)
        print("decoding numbers")
        for j in range(max_length_numbers):

            trg_tensor = torch.LongTensor(output_indices_numbers).unsqueeze(0).to(device)
            trg_mask = model.generate_square_subsequent_mask(len(trg_tensor)).to(device)

            '''
            ### --------------------- put this in the model!!! -----------------------------
            # add 0 feature to the encoded image
            image_memory = enc_src[..., None]
            image_memory = torch.cat([image_memory, torch.zeros(image_memory.shape).to(device)], dim=2)

            # add 1 feature to the target script
            script_memory = torch.tensor(output_indices_corpus)[..., None].to(device)
            script_memory = script_memory[None, ...] # batch
            script_memory = torch.cat([script_memory, torch.ones(script_memory.shape).to(device)], dim=2)

            # concatenate
            memory_enc_nbrs = torch.cat([
                image_memory, 
                script_memory, 
                torch.zeros((image_memory.shape[0], 1024 - image_memory.shape[1] - script_memory.shape[1], image_memory.shape[2])).to(device)], dim=1)
            ### ------------------------------------------------------------------------------
            '''

            with torch.no_grad():
                #output = model.forward_Numbers_Decoder(trg_tensor, memory_enc_nbrs, trg_mask)
                output = model.forward_Numbers_Decoder(trg_tensor, enc_src, corpus_output, model.generate_square_subsequent_mask(len(trg_tensor)).to(device))

            prediction = output[0,-1,0].item()
            iterator += 1
            if j < len(targets_numbers):
                ground_truth = targets_numbers[j]
            else:
                ground_truth = 0.0
            print("{:.2f}% -> Prediction:\t{:.4f}\tGroundtruth:\t{:.4f}\tAbs:\t{:.4f}".format(iterator / (max_length_encoding + max_length_numbers) * 100, prediction, ground_truth, abs(prediction-ground_truth)))


            output_indices_numbers.append(prediction)

        with open(os.path.join(output_folder, "output_" + str(iteration) + ".py"), "w+") as out_file:
            out_file.write(st.decode_encoded_script(output_indices_corpus, output_indices_numbers, encoder))
        with open(os.path.join(output_folder, "orig_" + str(iteration) + ".py"), "w+") as out_file:
            out_file.write(st.decode_encoded_script(targets_script, targets_numbers, encoder))
        save_dataset_render(image.cpu(), os.path.join(output_folder, "input_" + str(iteration) + ".png"))


if __name__ == "__main__":
    device = torch.device('cuda')
    encoding_file = "encoding.txt"
    dataset_path = "/home/baldur/Dataset/ShapeNet/Dataset_Polygen/Test"
    csv_path = os.path.join(dataset_path, "dataset.csv")
    model_folder = os.path.join(dataset_path, "models")
    model_path = os.path.join(model_folder, "/home/baldur/Nextcloud_private/Bachelorarbeit/Trained/ScriptGen/21_08_31/SkriptGen-Ep9")
    output_folder = "InferenceTesting"

    #encoding, len_script, len_numbers = st.load_sentence_encoding(encoding_file)

    ds = Dataset_ScriptGen(csv_path, dataset_path, None, None, None, transforms.Compose([
            Rescale(224, 224),
            ToTensor()
    ]))

    data = DataLoader(ds, 1, True)

    model = torch.load(model_path, map_location=device)
    model.eval()

    testModel(model, data, device, encoding_file, output_folder)
