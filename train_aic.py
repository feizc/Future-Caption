import argparse 
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch 
from tqdm import tqdm 
from torch.nn import functional as nnf 

from models import AICModel, TransformerConfig
from dataset import ClipCocoDataset 
from torch.utils.data import Dataset, DataLoader

use_device = torch.cuda.is_available()
device = torch.device('cuda:0' if use_device else 'cpu') 




def train(model, train_dataloader, args): 
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    ) 
    for epoch in range(args.epochs): 
        print(f">>> Training epoch {epoch}") 
        progress = tqdm(total=len(train_dataloader), desc='AICModel') 
        for idx, (tokens, _, img_features) in enumerate(train_dataloader):  
            model.zero_grad() 
            tokens, img_features = tokens.to(device), img_features.to(device, dtype=torch.float32) 
            outputs = model(img_features, tokens) 
            loss = nnf.cross_entropy(outputs.reshape(-1, outputs.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/train.pkl') 
    parser.add_argument('--tokenizer_path', default='./ckpt/gpt2') 
    parser.add_argument('--batch_size', default=5) 
    parser.add_argument('--lr', default=1e-2) 
    parser.add_argument('--epochs', default=10) 
    parser.add_argument('--warmup_steps', default=5000) 
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 
    dataset = ClipCocoDataset(args.data_path, tokenizer) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    config = TransformerConfig()
    model = AICModel(config).to(device)
    train(model, train_dataloader, args) 





if __name__ == '__main__': 
    main() 
