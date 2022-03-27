import paddle
import paddlenlp
import random
import numpy as np
random.seed(2323)
paddle.seed(2323)
np.random.seed(2323)
paddle.set_device('gpu')
rank = paddle.distributed.get_rank()
if paddle.distributed.get_world_size() > 1:
    paddle.distributed.init_parallel_env()
    
from paddlenlp.datasets import load_dataset
import pandas as pd
from tqdm import tqdm
def read_train_data(file_path):
    train_data = pd.read_csv(file_path, delimiter='\t')
    for index, items in tqdm(train_data.iterrows()):
        query, title, neg_title = items['query'], items['title'], items['neg_title']
        yield {'query':query, 'title':title, 'neg_title':neg_title}
               
def read_dev_data(file_path):
    dev_data = pd.read_csv(file_path, delimiter='\t')
    for index, items in tqdm(dev_data.iterrows()):
        query, title, label = items['query'], items['title'], items['label']
        yield {'query':query, 'title':title, 'label':label}

train_ds = load_dataset(read_train_data, file_path='../sort/train_pairwise.csv', lazy=False)
dev_ds = load_dataset(read_dev_data, file_path='../sort/dev_pairwise.csv', lazy=False)

model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')

def convert_example(example, 
                    tokenizer, 
                    max_seq_len, 
                    pad_to_max_seq_length= False, 
                    mode= 'train'):
        query, title= example['query'], example['title']     
        pos = tokenizer(
            text= query,
            text_pair= title,
            max_seq_len= max_seq_len,
            pad_to_max_seq_len= pad_to_max_seq_length
        )
        pos_input_ids, pos_token_type_ids = pos['input_ids'], pos['token_type_ids']
        
        if mode == 'train':
            neg_title = example['neg_title']            
            neg = tokenizer(
                text= query,
                text_pair= neg_title,
                max_seq_len=  max_seq_len,
                pad_to_max_seq_len= pad_to_max_seq_length
            )
            neg_input_ids, neg_token_type_ids = neg['input_ids'], neg['token_type_ids']
            return [pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids]
        
        else:
            if mode == 'dev':
                label = example['label']
                return [pos_input_ids, pos_token_type_ids, label]
            
            elif mode == 'predict':
                return [pos_input_ids, pos_token_type_ids]
            
            else:
                raise ValueError('not supported mode:{}'.format(mode))

from functools import partial
train_tran_func = partial(
    convert_example,
    tokenizer= tokenizer,
    max_seq_len= 128
)

dev_tran_func = partial(
    convert_example,
    tokenizer= tokenizer,
    max_seq_len= 128,
    mode= 'dev'
)    

from paddlenlp.data import Tuple, Pad, Stack
batchify_fn_train = lambda sample, fn = Tuple(
    Pad(axis= 0, pad_val= tokenizer.pad_token_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_type_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_type_id)
): [data for data in fn(sample)]

batchify_fn_dev = lambda sample, fn = Tuple(
    Pad(axis= 0, pad_val= tokenizer.pad_token_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_type_id),
    Stack(dtype= 'int64')
): [data for data in fn(sample)]

def create_dataloader(dataset, 
                       batchify_fn= None, 
                       trans_fun= None, 
                       mode= 'train', 
                       batch_size= 256):
    if trans_fun:
        dataset = dataset.map(trans_fun)
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset= dataset,
            batch_size= batch_size,
            shuffle= shuffle
        )
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset= dataset,
            batch_size= batch_size,
            shuffle= shuffle
        )
    return paddle.io.DataLoader(
        dataset= dataset,
        batch_sampler= batch_sampler,
        collate_fn= batchify_fn,
        return_list= True
    )

train_data_loader = create_dataloader(
    dataset= train_ds,
    batchify_fn= batchify_fn_train,
    trans_fun= train_tran_func,
    batch_size= 32
)
dev_data_loader = create_dataloader(
    dataset= dev_ds,
    batchify_fn= batchify_fn_dev,
    trans_fun= dev_tran_func,
    mode= 'dev',
    batch_size= 32
)
i = 0
for x in train_data_loader:
    print(x)
    i += 1
    if i == 10:
        break
i = 0 
for x in dev_data_loader:
    print(x)
    i += 1
    if i == 10:
        break
from paddle import nn
import paddle.nn.functional as F
class PairwiseMaching(nn.Layer):
    def __init__(
        self,
        pretrained_model,
        dropout= None,
        margin= 0.1):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin
        self.similarity = nn.Linear(self.ptm.config['hidden_size'], 1)
        
    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(shape= [None, None], dtype= 'int64'),
        paddle.static.InputSpec(shape= [None, None], dtype= 'int64')
    ])
    def get_pooled_embedding(self, 
                             input_ids,
                             token_type_ids= None,
                             position_ids= None,
                             attention_mask= None):
        _, cls_embedding = self.ptm(
            input_ids, 
            token_type_ids, 
            position_ids, 
            attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        sim = self.similarity(cls_embedding)
        return sim
    
    def predict(self,
                input_ids,
                token_type_ids= None,
                position_ids= None,
                attention_mask= None):       
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)
        sim_score = self.similarity(cls_embedding)
        sim_score = F.sigmoid(sim_score)

        return sim_score
    
    def forward(self,
                pos_input_ids,
                neg_input_ids,
                pos_token_type_ids= None,
                pos_position_ids= None,
                pos_attention_mask= None,
                neg_token_type_ids= None,
                neg_position_ids= None,
                neg_attention_mask= None):
        _, pos_cls_embedding = self.ptm(pos_input_ids, pos_token_type_ids,
                                        pos_position_ids, pos_attention_mask)

        _, neg_cls_embedding = self.ptm(neg_input_ids, neg_token_type_ids,
                                        neg_position_ids, neg_attention_mask)

        pos_embedding = self.dropout(pos_cls_embedding)
        neg_embedding = self.dropout(neg_cls_embedding)

        pos_sim = self.similarity(pos_embedding)
        neg_sim = self.similarity(neg_embedding)

        pos_sim = F.sigmoid(pos_sim)
        neg_sim = F.sigmoid(neg_sim)

        labels = paddle.full(
        shape=[pos_cls_embedding.shape[0]], fill_value=1.0, dtype='float32')

        loss = F.margin_ranking_loss(
        pos_sim, neg_sim, labels, margin=self.margin)
        
        return loss
model = PairwiseMaching(model, margin= 0.1)

epochs = 3
total_steps = epochs * len(train_data_loader)
lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(
    learning_rate= 2e-5,
    total_steps= total_steps,
    warmup= 0.1
)

decay_params = [p.name for n, p  in model.named_parameters() 
                if not any(nd in n for nd in ('bias', 'norm'))]

optimizer = paddle.optimizer.AdamW(
    learning_rate= 2e-5,
    parameters= model.parameters(),
    apply_decay_param_fun= lambda x: x in decay_params
)
metric = paddle.metric.Auc()  

@paddle.no_grad()
def evaluate(model, metric, data_loader, global_step, phase= 'dev'):
    model.eval()
    metric.reset()
    for idx, batch in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch
        pos_probs = model.predict(input_ids, token_type_ids)
        neg_probs = 1-pos_probs
        preds = np.concatenate((neg_probs, pos_probs), axis= 1)
        metric.update(preds= preds, labels= labels)
    print('global_step:{}, eval_{}, auc:{:.3}'.format(global_step, phase, metric.accumulate()))
    metric.reset()
    model.train()
    
global_step = 0
import time
import os
tic_train =  time.time()
for epoch in range(1, epochs + 1):
    for step, batch_data in enumerate(train_data_loader):
        pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids = batch_data
        loss = model(
            pos_input_ids= pos_input_ids,
            neg_input_ids= neg_input_ids,
            pos_token_type_ids= pos_token_type_ids,
            neg_token_type_ids= neg_token_type_ids
        )
        global_step += 1
        if global_step % 10 == 0 and rank == 0:
            print('global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s'
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        loss.backward()
        lr_scheduler.step()
        optimizer.step()
        optimizer.clear_grad()
        if global_step % 1000 == 0 and rank == 0:
            evaluate(model, metric, dev_data_loader, global_step, 'dev')
            
        if global_step % 5000 == 0 and rank == 0:
            save_dir = os.path.join('checkpoints', 'model_%d' % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_dir)
