import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Print working directory
print(sys.path)
from pathlib import Path

import torch
from transformers import BertTokenizer,GPT2LMHeadModel,GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
# from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
import pytorch_lightning as pl
import json
import argparse
import random
import warnings
from model import MLModel
from config import Config
from Matrics import NLP_Gen_Matrics
from pytorch_lightning.loggers import TensorBoardLogger
warnings.filterwarnings("ignore")





class Net(pl.LightningModule):
    def __init__(
        self,
        batch_size,
        epochs,
        t_total=100000,
        data_path=[],
        max_length=512,
        warm_up_steps=0,
        lr=1e-4,
        check_test = False,
        model_file = 'None',
    ):
        super(Net, self).__init__()
        self.check_test = check_test

        self.batch_size = batch_size
        self.epochs = epochs
        # The total number
        self.t_total = t_total
        # Preheat steps
        self.warm_up_steps = warm_up_steps
        self.lr = lr
        
        # self.bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
        # self.gpt_tok.pad_token = self.gpt_tok.eos_token
        
        self.config = Config()
        self.config.model_file = model_file
        self.model_file = model_file
        self.model = MLModel(self.config)
        self.max_length = max_length
        self.data_path = data_path
        self.real_data_num = 0
        self.g_step = 0
        
        self.data_language = self.config.data_language

        if self.data_language=='EN':
            # english_data
            with open(os.getcwd() + self.data_path[0] + "_train.txt",encoding='utf-8') as f:
                data_single = f.read()
            data_single_train = data_single.split('\n[SEP]\n')
            self.real_data_num = len(data_single_train)

            
        else:
            with open(os.getcwd() + self.data_path[1] + "_train.txt",encoding='utf-8') as f:
                data_single_train = f.read()
            data_single_train = data_single_train.replace('[SEP]','[SPE]')    
            data_single_train = data_single_train.replace('<|endoftext|>','[SEP]')
            data_single_train = data_single_train.split('\n[SPE]\n')

            self.real_data_num = len(data_single_train)
            
        
        # cuda
        self.model.to(self.device)
    def forward(self,inputs,is_train):
        outputs = self.model(inputs=inputs,is_train=is_train)
        return outputs
        
    # Loss function calculation
    def compute_loss(self,outputs):
        def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):  # [batch, latent]
            """ Kl divergence between two gaussian distribution formula """
            kld = 0.5 * torch.sum(prior_logvar - recog_logvar - 1
                                + recog_logvar.exp() / prior_logvar.exp()
                                + (prior_mu - recog_mu).pow(2) / prior_logvar.exp(), 1)
            return kld  # [batch]

        # get GPT - 2 losses and CVAE variance
        # prior average, log variance, posterior mean, log variance
        _mu, _logvar, mu, logvar,_mu_p_list, _logvar_p_list, mu_p_list, logvar_p_list,decode_loss,di_loss = outputs 

        # Kl divergence losses (batch)
        kld_loss = gaussian_kld(mu, logvar, _mu, _logvar)
        kld_loss_p_list = []
        for i in range(len(_mu_p_list)):
            kld_loss_p_list.append(gaussian_kld(mu_p_list[i], logvar_p_list[i], _mu_p_list[i], _logvar_p_list[i]))
        
        # Choose the biggest loss
        # kld_loss_p = torch.stack(kld_loss_p_list,dim=1).max(dim=1)[0]
        # The average loss
        kld_loss_p = torch.stack(kld_loss_p_list,dim=1).mean(dim=1)
        
        kld_loss = kld_loss + kld_loss_p
        kld_loss = kld_loss.mean()
        
        # Kl annealing
        kld_weight = 1
        # kld_weight = min(1.0 * self.g_step / self.warm_up_steps, 1)  # 一次性退火
        kld_weight = min(1.0 * (self.g_step % (2*self.warm_up_steps)) / self.warm_up_steps, 1)  # 周期性退火
        self.g_step += 1
        

        # loss
        # Multitasking loss weight
        kld_loss_w = 1
        di_loss_w = 1
        if self.config.conditions['no_bert'] or self.config.conditions['no_kl']:
            kld_loss_w = 0
            di_loss_w = 0
        
        if self.config.conditions['no_di']:
            di_loss_w = 0
            
        # di_loss_t = di_loss / torch.min(decode_loss,kld_loss,di_loss)
        loss = decode_loss + kld_weight * kld_loss * kld_loss_w
        di_loss = di_loss_w * di_loss

        return loss, decode_loss, kld_loss, di_loss

    # Data processing in advance
    def prepare_data(self):
        # The current working directory
        print(os.getcwd())
        if self.data_language=='EN':
            # english_data
            with open(os.getcwd() + self.data_path[0] + "_train.txt",encoding='utf-8') as f:
                data_single = f.read()
            data_single_train = data_single.split('\n[SEP]\n')
            random.shuffle(data_single_train)

            
            with open(os.getcwd() + self.data_path[0] + "_test.txt",encoding='utf-8') as f:
                data_single = f.read()
            data_single_test = data_single.split('\n[SEP]\n')
            random.shuffle(data_single_test)
            
            eos_token = '<|endoftext|>'
            
        else:
            with open(os.getcwd() + self.data_path[1] + "_train.txt",encoding='utf-8') as f:
                data_single_train = f.read()
            data_single_train = data_single_train.replace('[SEP]','[SPE]')    
            data_single_train = data_single_train.replace('<|endoftext|>','[SEP]')
            data_single_train = data_single_train.split('\n[SPE]\n')

            
            with open(os.getcwd() + self.data_path[1] + "_test.txt",encoding='utf-8') as f:
                data_single_test = f.read()
            data_single_test = data_single_test.replace('[SEP]','[SPE]')    
            data_single_test = data_single_test.replace('<|endoftext|>','[SEP]')
            data_single_test = data_single_test.split('\n[SPE]\n')

            
            eos_token = '[SEP]'
                
            
        
        data_train = []
        for line in data_single_train:
            temp = line.split('\n')
            temp[1] = eos_token + temp[1].split(eos_token)[-2] + eos_token
            # temp[2] = temp[2].replace(' <|endoftext|>','<|endoftext|>')
            if temp[1] == eos_token + eos_token:
                continue
            data_train.append(temp+[temp[1]+temp[2]])
        data_single_train = data_train
        
        
        data_test = []
        for line in data_single_test:
            temp = line.split('\n')
            temp[1] = eos_token + temp[1].split(eos_token)[-2] + eos_token
            # temp[2] = temp[2].replace(' <|endoftext|>','<|endoftext|>')
            if temp[1] == eos_token + eos_token:
                continue
            data_test.append(temp+[temp[1]+temp[2]])
        data_single_test = data_test
        
        if self.check_test:
            data_single_train = data_single_train[:1000]
            data_single_test = data_single_test[:30]
            
            
        random.shuffle(data_single_train)
        random.shuffle(data_single_test)
        
        
        self.dataset_train = data_single_train
        self.dataset_valid = data_single_test
        self.small_test = self.dataset_valid[-10:]

        self.real_data_num = len(self.dataset_train)
        self.t_total = self.epochs*self.real_data_num//self.batch_size
        self.warm_up_steps = self.t_total//5
        
        
        print('Data sample:',self.dataset_train[0])


    def collate_fn_text(self,batch):
        # batch Internal triples persona,x,y
        inputs = {'persons':[],'posts':[],'responses':[],'p_r':[]}
        for line in batch:
            inputs['persons'].append(line[0])
            inputs['posts'].append(line[1])
            inputs['responses'].append(line[2])
            inputs['p_r'].append(line[3])
            
        inputs['sampled_latents'] = torch.randn(len(batch), self.config.latent_dim)
        
        return inputs
            
    # The data load
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=self.collate_fn_text,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=self.collate_fn_text,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=self.collate_fn_text,
        )
        

    # The optimizer
    def configure_optimizers(self):
        print('学习率预热:',self.warm_up_steps,self.t_total)
        optimizer_g = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler_g = get_linear_schedule_with_warmup(
            optimizer_g, self.warm_up_steps, self.t_total
        )
        scheduler_g = {"scheduler": scheduler_g, "interval": "step", "frequency": 1}
        

        # temp_print = self.model.parameters()
        
        optimizer_d = AdamW([{'params':self.model.encoder.parameters()},{'params':self.model.cls.parameters()}], lr=self.lr/2, weight_decay=0.001)
        scheduler_d = get_linear_schedule_with_warmup(
            optimizer_d, self.warm_up_steps, self.t_total
        )
        scheduler_d = {"scheduler": scheduler_d, "interval": "step", "frequency": 1}
        

        return [optimizer_g,optimizer_d], [scheduler_g,scheduler_d]


    def training_step(self, batch, batch_nb, optimizer_idx):
        tensorboard = self.logger.experiment
        outputs = self.forward(batch,is_train=True)
        loss, decode_loss, kld_loss, di_loss = self.compute_loss(outputs)
        loss_dict = {'train_loss': loss, 'decode_loss': decode_loss, 'kld_loss': kld_loss, 'di_loss': di_loss}
        # print(decode_loss, kld_loss, di_loss)
        self.log(
            "train_loss",
            loss+di_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # for key in loss_dict:
        #     self.log(key, loss_dict[key], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tensorboard.add_scalars('loss_dict',loss_dict,self.global_step)
        
        if optimizer_idx == 0:
            return loss
        else:
            return di_loss

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch,is_train=True)
        loss, decode_loss, kld_loss, di_loss = self.compute_loss(outputs)
        loss_dict = {'val_loss': loss, 'val_decode_loss': decode_loss, 'val_kld_loss': kld_loss, 'val_di_loss': di_loss}
        return loss_dict
    
    def validation_epoch_end(self, outputs):
        # On average, the output is a list of each element in the list of validation_step return values
        val_loss_list = []
        val_decode_loss_list = []
        val_kld_loss_list = []
        val_di_loss_list = []
        for i in range(len(outputs)):
            val_loss_list.append(outputs[i]['val_loss'])
            val_decode_loss_list.append(outputs[i]['val_decode_loss'])
            val_kld_loss_list.append(outputs[i]['val_kld_loss'])
            val_di_loss_list.append(outputs[i]['val_di_loss'])
        
        # averaging
        avg_val_loss = torch.stack(val_loss_list).mean()
        avg_val_decode_loss = torch.stack(val_decode_loss_list).mean()
        avg_val_kld_loss = torch.stack(val_kld_loss_list).mean()
        avg_val_di_loss = torch.stack(val_di_loss_list).mean()
        
        self.log(
            "avg_val_loss",
            avg_val_loss+avg_val_di_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )       
        
        self.log(
            'avg_val_decode_loss',
            avg_val_decode_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            'avg_val_kld_loss',
            avg_val_kld_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            'avg_val_di_loss',
            avg_val_di_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
            
        
    def test_step(self, batch, batch_nb):
        persona_sequences,context_sequences,output_sequences,gold_sequences = self.forward(batch,is_train=False)
        return {'persona':persona_sequences,'context':context_sequences,'pred':output_sequences,'target':gold_sequences}
    
    def test_epoch_end(self, outputs) -> None:
        persona_sequences = []
        context_sequences = []
        output_sequences = []
        gold_sequences = []
        for i in range(len(outputs)):
            persona_sequences += outputs[i]['persona']
            context_sequences += outputs[i]['context']
            output_sequences += outputs[i]['pred']
            gold_sequences += outputs[i]['target']
        
        # save the result
        # format: 'I [CSE] am [CSE] you \ n my [CSE] is [CSE] he \ n \ n'
        
        pred_list = []
        for i in range(len(output_sequences)):
            pred_list.append('[CSE]'.join(persona_sequences[i])+ '\n' + '[CSE]'.join(context_sequences[i]) + '\n'+  '[CSE]'.join(output_sequences[i])+ '\n'+ '[CSE]'.join(gold_sequences[i]))

        
        with open(os.getcwd() + './result/' + self.model_file + '-pred_result.txt','w',encoding='utf-8') as f:
            f.write('\n\n'.join(pred_list))
        
        
        

if __name__ == "__main__":
    config = Config()
    # seed
    pl.seed_everything(config.seed)
    check_test = False
    
    model_name = 'CLVGen'
    data_language = config.data_language
    model_file = model_name + '_' + data_language
    for key in config.conditions.keys():
        if config.conditions[key]:
            model_file = model_file + '_' + key
    print('model_file:',model_file)
    config.model_file = model_file
    
    print(config.conditions)
    

    max_length = config.max_len
    batch_size = config.batch_size
    epochs = config.epochs
    output_path = config.output_dir
    lr = config.lr
    data_path = config.data_path

    
    logger = TensorBoardLogger('tb_logs', name=model_file)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        save_top_k=1,
        monitor="avg_val_loss",
        mode="min",
        save_weights_only = True,
        filename= model_file + '-{epoch:02d}-{step}-{avg_val_loss:.2f}',
    )
    learning_rate_callback = LearningRateMonitor()
    
    early_stopping = EarlyStopping(
        monitor='avg_val_loss', 
        min_delta=0.0, 
        patience=3, 
        mode='min', 
        strict=True
    )

    net = Net(
        batch_size,
        epochs,
        data_path=data_path,
        max_length=max_length,
        lr=lr,
        check_test = check_test,
        model_file = model_file,
    )
    data_nums = net.real_data_num
    # The total number
    t_total = epochs*data_nums//batch_size
    # The eval steps
    if data_language == 'EN':
        eval_interval = int(2441*(8/batch_size))
    else:
        eval_interval = int(3125*(8/batch_size))
    if check_test:
        eval_interval = 10
    if  t_total != 0:
        eval_interval = t_total//10
       
    trainer = pl.Trainer(
        default_root_dir=output_path,
        gradient_clip_val=0.5,
        max_epochs=epochs,
        gpus=1,
        val_check_interval=eval_interval,
        callbacks=[learning_rate_callback, checkpoint_callback,early_stopping],
        logger = logger,
    )    
    
    trainer.fit(net)
    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)
    d = torch.load(best_model_path)["state_dict"]
    #trainer.save_checkpoint('latest-'+best_model_path)
    print('best_model_path:',best_model_path)
    
    # If you want to reasoning, can load model
    # d = torch.load('./model/CLVGen.ckpt')["state_dict"]
    
    net.load_state_dict(d, strict=False)    
    trainer.test(model = net)
    
    judger = NLP_Gen_Matrics(model_name=model_file,condition_list=['2023-01-01'])
    result = judger.get_judge_data('./result/' + config.model_file + '-pred_result.txt')
    # net.predict_test()
