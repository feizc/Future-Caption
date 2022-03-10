import argparse 
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch 
from tqdm import tqdm 
from torch.nn import functional as nnf 
import os 

from models import AICModel, TransformerConfig
from dataset import ClipCocoDataset 
from torch.utils.data import Dataset, DataLoader
import evaluation 
from evaluation import PTBTokenizer, Cider

use_device = torch.cuda.is_available()
device = torch.device('cuda:0' if use_device else 'cpu') 
torch.backends.cudnn.benchmark = True



def evaluate_metrics(model, test_dataloader, tokenizer, epoch): 
    import itertools
    model.eval() 
    gen = {} 
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_dataloader)) as pbar:
        for idx, (tokens, _, img_features) in enumerate(test_dataloader):  
            img_features = img_features.to(device) 
            with torch.no_grad():
                text, _ = model.beam_search(img_features, beam_size=5, out_size=1) 

            caps_gt = tokenizer.batch_decode(tokens)
            caps_gen = tokenizer.batch_decode(text)

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (idx, i)] = [gen_i, ]
                gts['%d_%d' % (idx, i)] = gts_i
            pbar.update()
            break
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_all_scores(gts, gen) 
    print(scores)
    return scores




def train_xe(model, train_dataloader, args, optimizer, scheduler, epoch): 
    model.train()
    
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
        break 
    progress.close()
    torch.save(
        model.state_dict(), 
        os.path.join(args.out_dir, f"{args.model_type}-{epoch:02d}.pt")
    )
    




def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/train.pkl') 
    parser.add_argument('--tokenizer_path', default='./ckpt/gpt2') 
    parser.add_argument('--batch_size', default=5) 
    parser.add_argument('--lr', default=1e-2) 
    parser.add_argument('--epochs', default=10) 
    parser.add_argument('--warmup_steps', default=5000) 
    parser.add_argument('--out_dir', default='./ckpt') 
    parser.add_argument('--model_type', default='aic') 
    parser.add_argument('--phase', type=str, default='xe', choices=('xe', 'scst'))
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 
    dataset = ClipCocoDataset(args.data_path, tokenizer) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    config = TransformerConfig()
    model = AICModel(config).to(device) 

    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    ) 
    for epoch in range(args.epochs): 
        train_xe(model, train_dataloader, args, optimizer, scheduler, epoch) 
        evaluate_metrics(model, train_dataloader, tokenizer, epoch)
        break 





if __name__ == '__main__': 
    main() 
