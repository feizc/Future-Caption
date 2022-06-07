import torch 
from models import NAICModel, AICModel 
from transformers import GPT2Tokenizer, AdamW 
import torch.nn as nn 
import random 
import numpy as np 
from dataset import ClipCocoDataset 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from torch.nn import functional as nnf 
import itertools
import evaluation 
from evaluation import PTBTokenizer, Cider 
import os 
import argparse

from models.configure import TransformerConfig 
from train_aic import evaluate_metrics, train_scst 

use_device = torch.cuda.is_available()
device = torch.device('cuda:0' if use_device else 'cpu') 
torch.backends.cudnn.benchmark = True

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234) 

SPECIAL_TOKENS = ["<mask>"] 
SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<mask>"]} 



def confidence_selection(logits, threshold=0.60): 
    max_logits, _ = torch.max(logits, -1) 
    less_threshold_idx = torch.nonzero(max_logits < threshold)
    # print(max_logits, less_threshold_idx) 
    return less_threshold_idx.squeeze(1) 



def train(student_model, teacher_model, train_dataloader, args, optimizer, epoch, tokenizer): 
    student_model.train() 
    teacher_model.eval() 
    running_loss = .0 
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    progress = tqdm(total=len(train_dataloader), desc='combine') 
    alpha = float((args.epochs - epoch)/ args.epochs)
    for idx, (tokens, _, img_features) in enumerate(train_dataloader):  
        student_model.zero_grad() 
        tokens, img_features = tokens.to(device), img_features.to(device, dtype=torch.float32) 
        student_outputs = student_model(img_features, tokens) 
        ce_loss = nnf.cross_entropy(student_outputs.reshape(-1, student_outputs.shape[-1]), tokens.flatten())
        
        less_threshold_idx = confidence_selection(student_outputs.squeeze(0).clone().detach()) 
        teacher_inputs = tokens.clone().squeeze(0).detach()
        teacher_inputs[less_threshold_idx] = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)[0]
        teacher_inputs = teacher_inputs.unsqueeze(0)
        with torch.no_grad():
            teacher_outputs = teacher_model(img_features, teacher_inputs) 
        teacher_outputs = torch.index_select(teacher_outputs, 1, less_threshold_idx).reshape(-1, teacher_outputs.shape[-1])
        student_outputs = torch.index_select(student_outputs, 1, less_threshold_idx).reshape(-1, student_outputs.shape[-1]) 
        kl_loss = kl_loss_fn(student_outputs, teacher_outputs) 
        
        loss = (1 - alpha) * ce_loss + alpha * kl_loss 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        progress.set_postfix({"loss": running_loss / (idx + 1)})
        progress.update()

        break  
    progress.close()
    return running_loss / len(train_dataloader)



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/train.pkl') 
    parser.add_argument('--tokenizer_path', default='./ckpt/gpt2') 
    parser.add_argument('--batch_size', default=1) 
    parser.add_argument('--lr', default=1e-3) 
    parser.add_argument('--epochs', default=10) 
    parser.add_argument('--warmup_steps', default=5000) 
    parser.add_argument('--out_dir', default='./ckpt') 
    parser.add_argument('--model_type', default='combine') 
    args = parser.parse_args()  

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 
    # add mask token to vocabulary 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    dataset = ClipCocoDataset(args.data_path, tokenizer, padding=False) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    
    config = TransformerConfig(vocab_size=len(tokenizer)) 
    teacher_model = NAICModel(config).to(device) 
    student_model = AICModel(config).to(device) 

    optimizer = AdamW(student_model.parameters(), lr=args.lr)  

    best_cider = .0 
    for epoch in range(args.epochs):
        loss = train(student_model, teacher_model, train_dataloader, args, optimizer, epoch, tokenizer) 
        # As original model is kept unchanged, the evaluation is reusable 
        scores = evaluate_metrics(student_model, train_dataloader, tokenizer, epoch) 
        val_cider = scores['CIDEr'] 

        if val_cider >= best_cider:
            best_cider = val_cider
            torch.save(
                student_model.state_dict(), 
                os.path.join(args.out_dir, f"{args.model_type}-{epoch:02d}.pt")
            )
        break 



if __name__ == '__main__':  
    main()
