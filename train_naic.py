import torch 
from models import NAICDecoder, Encoder, NAICModel 
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
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

use_device = torch.cuda.is_available()
device = torch.device('cuda:0' if use_device else 'cpu') 
torch.backends.cudnn.benchmark = True

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234) 

SPECIAL_TOKENS = ["<mask>"] 
SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<mask>"]} 



# mask sentence for generation, referring from huggingface 
def mask_tokens(inputs, tokenizer, mask_probability): 
    labels = inputs.clone() 
    masked_indices = torch.bernoulli(torch.full(labels.shape, mask_probability)).bool() 
    labels[~masked_indices] = -1 

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)[0]
    return inputs, labels



def train(model, train_dataloader, args, optimizer, scheduler, epoch, tokenizer): 
    model.train() 
    running_loss = .0 
    progress = tqdm(total=len(train_dataloader), desc='NAICModel') 
    for idx, (tokens, _, img_features) in enumerate(train_dataloader):  
        model.zero_grad() 
        tokens, img_features = tokens.to(device), img_features.to(device, dtype=torch.float32) 
        inputs, labels = mask_tokens(tokens, tokenizer, args.mask_probability) 
        outputs = model(img_features, inputs) 
        loss = nnf.cross_entropy(outputs.reshape(-1, outputs.shape[-1]), labels.flatten(), ignore_index=-1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        progress.set_postfix({"loss": running_loss / (idx + 1)})
        progress.update()
        break 
    progress.close()
    return running_loss / len(train_dataloader)



def evaluate_metrics(model, test_dataloader, tokenizer, epoch): 
    model.eval() 
    gen = {} 
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_dataloader)) as pbar:
        for idx, (tokens, _, img_features) in enumerate(test_dataloader):  
            tokens, img_features = tokens.to(device), img_features.to(device, dtype=torch.float32) 
            with torch.no_grad(): 
                inputs, labels = mask_tokens(tokens, tokenizer, 1.0) 
                logits = model(img_features, inputs)
            gen_idx = torch.argmax(logits, -1)
    
            caps_gt = tokenizer.batch_decode(tokens)
            caps_gen = tokenizer.batch_decode(gen_idx)

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



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/train.pkl') 
    parser.add_argument('--tokenizer_path', default='./ckpt/gpt2') 
    parser.add_argument('--batch_size', default=1) 
    parser.add_argument('--lr', default=1e-2) 
    parser.add_argument('--epochs', default=10) 
    parser.add_argument('--warmup_steps', default=5000) 
    parser.add_argument('--out_dir', default='./ckpt') 
    parser.add_argument('--model_type', default='naic') 
    parser.add_argument('--mask_probability', default=0.80) 
    args = parser.parse_args() 

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path) 
    # add mask token to vocabulary 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    dataset = ClipCocoDataset(args.data_path, tokenizer, padding=False) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    
    #print(tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))

    config = TransformerConfig(vocab_size=len(tokenizer)) 
    model = NAICModel(config).to(device) 

    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    ) 

    for epoch in range(args.epochs): 
        train_loss = train(model, train_dataloader, args, optimizer, scheduler, epoch, tokenizer) 
        scores = evaluate_metrics(model, train_dataloader, tokenizer, epoch)
        
        torch.save(
            model.state_dict(), 
            os.path.join(args.out_dir, f"{args.model_type}-{epoch:02d}.pt")
        )
        break 




if __name__ == '__main__':   
    main()
    '''
    input_f = torch.randn((5,16,512)) 
    input_l = torch.ones((5,20)).long()
    v_encoder = Encoder()
    l_decoder = NAICDecoder() 
    enc_o = v_encoder(input_f)
    print(enc_o[0].size())
    dec_o = l_decoder(input_l, enc_o[0], None)
    print(dec_o[0].size())
    '''