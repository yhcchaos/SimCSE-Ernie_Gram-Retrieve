import paddle
import paddlenlp
paddle.set_device('cpu')
rank = paddle.distributed.get_rank()
if paddle.distributed.get_world_size() >= 1:
    paddle.distributed.init_parallel_env()
paddle.seed(4321)
import random
random.seed(4321)
import numpy as np
np.random.seed(4321)

def read_next_pair(file_path):
    with open(file_path, encoding= 'utf-8') as f:
        for line in f:
            data = line.rstrip().split('\t')
            if len(data) != 2:
                continue
            else:
                text1 = data[0]
                text2 = data[1]
                yield {'text1': text1, 'text2':text2}
from paddlenlp.datasets import load_dataset
dataset = load_dataset(read_next_pair, file_path= '../recall/train.csv', lazy= False)

import paddlenlp.transformers as transformers
tokenizer = transformers.ErnieTokenizer.from_pretrained('ernie-1.0')
model = transformers.ErnieModel.from_pretrained('ernie-1.0')
def convert_example(example, tokenizer, max_seq_len, pad_to_max_seq_len= False):
    result = []
    for key, text in example.items():
        encoded_input = tokenizer(
            text= text,
            max_seq_len = max_seq_len,
            pad_to_max_seq_len = pad_to_max_seq_len
        )
        input_ids = encoded_input['input_ids']
        input_type_ids = encoded_input['token_type_ids']
        result += [input_ids, input_type_ids]
    return result
from functools import partial
trans_func = partial(
    convert_example,
    tokenizer= tokenizer,
    max_seq_len= 60
)
from paddlenlp.data import Pad, Tuple
batchify_fn = lambda sample, fn = Tuple(
    Pad(axis= 0, pad_val= tokenizer.pad_token_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_type_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_id),
    Pad(axis= 0, pad_val= tokenizer.pad_token_type_id)
): [data for data in fn(sample)]
def create_dataloader(dataset, batchify_fn= None, batch_size= 256, trans_func= None, mode= 'train'):
    if trans_func:
        dataset = dataset.map(trans_func)
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset= dataset,
            shuffle= shuffle,
            batch_size= batch_size
        )
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset= dataset,
            shuffle= shuffle,
            batch_size= batch_size
        )
    return paddle.io.DataLoader(
        dataset= dataset,
        batch_sampler= batch_sampler,
        collate_fn= batchify_fn,
        return_list= True
    )

train_data_loader = create_dataloader(
    dataset= dataset,
    batchify_fn= batchify_fn,
    batch_size= 64,
    trans_func= trans_func,
    mode= 'train'
)
from paddle import nn
import paddle.nn.functional as F
import abc
class SemanticIndexBase(nn.Layer):
    def __init__(self, pretrained_model, dropout= None, output_emb_size= None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer= nn.initializer.TruncatedNormal(std= 0.02)
            )
            self.reduce_emb_layer = nn.Linear(768, output_emb_size,weight_attr= weight_attr)

    @paddle.jit.to_static(input_spec=[
        paddle.static.InputSpec(
            shape= [None, None], dtype= 'int64'
        ),
        paddle.static.InputSpec(
            shape= [None, None], dtype= 'int64'
        )
    ])

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids= None,
                             input_position_ids= None,
                             attention_mask= None):
        print("sfadfaf")
        print(input_ids, token_type_ids)
        _, cls_embedding = self.ptm(input_ids, token_type_ids, input_position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.reduce_emb_layer(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cla_embedding = F.normalize(cls_embedding, p= 2, axis= -1)
        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)
                batch_embeddings = self.get_pooled_embedding(input_ids, token_type_ids)
                yield batch_embeddings

    def cosin_sim(self,
                  query_input_ids,
                  title_input_ids,
                  query_token_type_ids= None,
                  query_position_ids= None,
                  query_attention_mask= None,
                  title_token_type_ids= None,
                  title_position_ids= None,
                  title_attention_mask= None):

        query_embedding = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask
        )

        title_embedding= self.get_pooled_embedding(
            title_input_ids,
            title_token_type_ids,
            title_position_ids,
            title_attention_mask
        )

        cosin_sim = paddle.sum(query_embedding * title_embedding, axis= -1)
        return cosin_sim
    @abc.abstractmethod
    def forward(self):
        pass
class SemanticIndexBatchNeg(SemanticIndexBase):
    def __init__(self,
                 pretrained_model,
                 dropout= None,
                 margin= 0.3,
                 scale= 30,
                 output_emb_size= None):
        super().__init__(pretrained_model, dropout, output_emb_size= output_emb_size)
        self.margin= margin,
        self.scale= scale

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids= None,
                query_position_ids= None,
                query_attention_mask= None,
                title_token_type_ids= None,
                title_position_ids= None,
                title_attention_mask= None):
        print(query_input_ids, query_token_type_ids)
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask
        )
        print("11111")
        print(title_input_ids, title_token_type_ids)
        title_cls_embedding= self.get_pooled_embedding(
            title_input_ids,
            title_token_type_ids,
            title_position_ids,
            title_attention_mask
        )

        cosin_sim= paddle.matmul(
            query_cls_embedding,
            title_cls_embedding,
            transpose_y= True
        )

        margin_diag = paddle.full(
            shape= [query_cls_embedding.shape[0]],
            fill_value= self.margin,
            dtype= paddle.get_default_dtype()
        )
        cosin_sim -= paddle.diag(margin_diag)
        cosin_sim *= self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype= 'int64')
        labels = paddle.reshape(labels, shape= [-1, 1])
        loss = F.cross_entropy(cosin_sim, label= labels)
        return loss
import os
model = SemanticIndexBatchNeg(
    pretrained_model= model,
    output_emb_size= 256,
    margin= 0.2
)
init_from_ckpt = None
if init_from_ckpt and os.path.isfile(init_from_ckpt):
    state_dict = paddle.load(init_from_ckpt)
    model.set_dict(state_dict)
    print("warm from:{}".format(init_from_ckpt))

# model = paddle.DataParallel(model)
epochs = 10
total_steps = epochs * len(train_data_loader)
lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(
    learning_rate= 5e-5,
    warmup= 0.1,
    total_steps= total_steps
)

decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ('bias', 'norm'))
]

optimizer = paddle.optimizer.AdamW(
    learning_rate= 5e-5,
    parameters= model.parameters(),
    apply_decay_param_fun= lambda x: x in decay_params
)
import time
glabal_steps= 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch_data in enumerate(train_data_loader, start= 1):
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch_data
        loss = model(
            query_input_ids= query_input_ids,
            title_input_ids= title_input_ids,
            query_token_type_ids= query_token_type_ids,
            title_token_type_ids= title_token_type_ids
        )

        global_steps += 1
        if global_steps % 10 == 0 and rank == 0:
            print('global step: %d, epoch %d, loss: %.5f, speed: %.2f step/s'
                  % (global_steps, epoch, loss, 10 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        if global_steps % 10 == 0 and rank == 0:
            save_dir = os.path.join('checkpoints', 'model_%d' % global_steps)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(save_param_path)
            tokenizer.save_pretrained(save_dir)

