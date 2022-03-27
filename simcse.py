import paddle
paddle.set_device('gpu')
rank = paddle.distributed.get_rank()
rank
if paddle.distributed.get_world_size() >= 1:
    paddle.distributed.init_parallel_env()
    
import random
random.seed(1234)
import numpy as np
np.random.seed(1234)
paddle.seed(1234)

import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset

def read_simcse_text(data_path):
    with open(data_path, encoding= 'utf-8') as f:
        for line in f:
            data = line.rstrip()
            yield {"text_a": data, "text_b": data}

train_ds = load_dataset(
        read_simcse_text, data_path= "../recall/train_unsupervised.csv", lazy= False
)

pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
    'ernie-1.0', 
    hidden_dropout_prob = 0.2, 
    attention_probs_dropout_prob= 0.2
)

tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

def convert_example(example, tokenizer, max_seq_length= 512, do_evaluate= False):
    result = []
    for  key, text in example.items():
        if 'label' in key:
            result += [example['label']]
        else:
            encoded_inputs = tokenizer(text= text, max_seq_len= max_seq_length)
            input_ids = encoded_inputs['input_ids']
            input_type_ids = encoded_inputs['token_type_ids']
            result += [input_ids, input_type_ids]
    return result

from functools import partial
trans_func = partial(
    convert_example,
    tokenizer= tokenizer,
    max_seq_length= 64
)
from paddlenlp.data import Tuple, Pad
batchify_fn = lambda samples, fn = Tuple(
    Pad(axis= 0, pad_val= tokenizer.pad_token_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_type_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_type_id)
): [data for data in fn(samples)]

from paddle import nn
import paddle.nn.functional as F
class SimCSE(nn.Layer):
    def __init__(self,
                pretrained_model,
                dropout= None,
                margin= 0.0,
                scale= 20,
                output_emb_size= None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer= nn.initializer.TruncatedNormal(std= 0.02)
            )
            self.emb_reduce_linear = nn.layer.Linear(
                768, output_emb_size, weight_attr= weight_attr
            )
        self.margin = margin
        self.scale = scale
        
    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(shape= [None, None], dtype= 'int64'),
        paddle.static.InputSpec(shape= [None, None], dtype= 'int64')
    ])
    
    def get_pooled_embedding(self,
                            input_ids,
                            input_token_type_ids= None,
                            input_position_ids= None,
                            input_attention_mask= None,
                            with_pool= True
                            ):
        sequence_output, cls_embedding = self.ptm(input_ids,
                                 input_token_type_ids,
                                 input_position_ids,
                                 input_attention_mask)
        if with_pool == False:
            cls_embedding = sequence_output[:, 0, :]
        
        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p= 2, axis= -1)
        return cls_embedding
    
    def get_semantic_embedding(self, dataloader):
        self.eval()
        with paddle.no_grad():
            for batch_data in dataloader:
                input_ids, input_type_ids = dataloader
                input_ids = paddle.to_tensor(input_ids)
                input_type_ids = paddle.to_tensor(input_type_ids)
                
                text_embedding= self.get_pooled_embedding(input_ids, input_token_type_ids= input_type_ids)
                yield text_embedding
    
    def cosine_sim(self,
                  query_input_ids,
                  title_input_ids,
                  query_token_type_ids= None,
                  query_position_ids= None,
                  query_attention_mask= None,
                  title_token_type_ids= None,
                  title_position_ids= None,
                  title_attention_mask= None,
                  with_pool= True
                  ):
        query_cls_embedding= self.get_pooled_embedding(query_input_ids,
                                                      query_input_ids,
                                                      query_position_ids,
                                                      query_attention_mask,
                                                      with_pool= with_pool
                                                      )
        title_cls_embedding= self.get_pooled_embedding(title_input_ids,
                                                      title_token_type_ids,
                                                      title_position_ids,
                                                      title_attention_mask
                                                      )
        cosine_sim= paddle.sum(query_cls_embedding * title_cls_embedding, axis= -1)
        return cosine_sim
        
    def forward(self,
               query_input_ids,
               title_input_ids,
               query_token_type_ids= None,
               query_position_ids= None,
               query_attention_mask= None,
               title_token_type_ids= None,
               title_position_ids= None,
               title_attention_mask= None
               ):
        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                       query_token_type_ids,
                                                       query_position_ids,
                                                       query_attention_mask
                                                       )
        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                       title_token_type_ids,
                                                       title_position_ids,
                                                       title_attention_mask
                                                       )
        cosine_sim = paddle.matmul(query_cls_embedding,
                                  title_cls_embedding,
                                  transpose_y= True
                                  )
        margin_diag = paddle.full(shape= [query_cls_embedding.shape[0]],
                                 fill_value= self.margin,
                                 dtype= paddle.get_default_dtype()
                                 )
        cosine_sim -= paddle.diag(margin_diag)
        
        cosine_sim *= self.scale
        lables= paddle.arange(0,
                              query_cls_embedding.shape[0],
                              dtype= 'int64')
        lables = paddle.reshape(lables, shape= [-1, 1])
        loss= F.cross_entropy(input= cosine_sim, label= lables)
        return loss

def create_dataloader(dataset,
                     mode= 'train',
                     batch_size= 1,
                     batchify_fn= None,
                     trans_fn= None):
    if trans_fn:
        dataset = dataset.map(trans_fn)
        
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset,
            batch_size= 64,
            shuffle= shuffle
        )
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                              batch_size= batch_size,
                                              shuffle= shuffle)
    return paddle.io.DataLoader(dataset= dataset,
                               batch_sampler= batch_sampler,
                               collate_fn= batchify_fn,
                               return_list= True)
    
train_data_loader = create_dataloader(train_ds,
                                     mode= 'train',
                                     batch_size= 64,
                                     batchify_fn= batchify_fn,
                                     trans_fn= trans_func)

model = SimCSE(pretrained_model,
               output_emb_size= 256
               )
model = paddle.DataParallel(model)
epochs = 3
num_training_steps = epochs * len(train_data_loader)
from paddlenlp.transformers import LinearDecayWithWarmup
lr_scheduler = LinearDecayWithWarmup(learning_rate= 5e-5, 
                                    total_steps= num_training_steps,
                                    warmup= 0.1
                                    )
decay_params = [
    p.name for n,p in model.named_parameters() if not any(nd in n for nd in ['bias', 'norm'])
]

optimizer = paddle.optimizer.AdamW(learning_rate= 5e-5,
                                  parameters= model.parameters(),
                                  weight_decay= 0.01,
                                  apply_decay_param_fun= lambda x: x in decay_params
                                  )

import time
start_time = time.time()
global_step = 0
tic_time = time.time()
from visualdl import LogWriter
writer = LogWriter(logdir= "./log/scalar_test/train")
import os
for epoch in range(1, 4):
    for step, batch in enumerate(train_data_loader, start= 1):
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch
        loss = model(query_input_ids= query_input_ids, 
                     title_input_ids= title_input_ids,
                     query_token_type_ids= query_token_type_ids,
                     title_token_type_ids= title_token_type_ids
                    )
        
        global_step += 1
        if global_step % 10 == 0 and rank == 0:
            print("global step {%d}, epoch: {%d}, batch: {%d}, loss: {%.5f}, speed: {%.2f} step/s" % (global_step, epoch, step, loss, 10 / (time.time() - tic_time)))
            writer.add_scalar(tag= "loss", step= global_step, value= loss)
            tic_time = time.time()
            
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
        if global_step % 2000 == 0 and rank == 0:           
            save_dir = os.path.join('checkpoints', "model_%d" % (global_step))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_dir)
time_end = time.time()
print("totally_cost: ". time_end - start_time)
        
