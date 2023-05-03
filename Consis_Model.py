import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import RobertaTokenizer, BertModel,RobertaForSequenceClassification
import random
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
import os
from config import Config
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Pl template
class Consis_Data(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset_train = None
        self.dataset_valid = None
        self.batch_size = 16
        config = Config()
        data_language = config.data_language        
        
        if data_language=='EN':
            self.tok = RobertaTokenizer.from_pretrained("xlm-roberta-base",do_lower_case=False)
        else:
            self.tok = RobertaTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.tok.sep_token = None
        
        # Add new words dont forget add in txt
        self.tok.add_tokens(['[PERSONA]','[CONTEXT]','[RESPONSE]'],special_tokens=True)


    def setup(self,stage):
        config = Config()
        data_language = config.data_language  
        if data_language=='EN':
        # english_data
            with open(os.getcwd() + self.data_path[0] + "_train.txt",encoding='utf-8') as f:
                data_single = f.read()
        else:
            with open(os.getcwd() + self.data_path[1] + "_train.txt",encoding='utf-8') as f:
                data_single = f.read()
        data_single = data_single.split('\n[SEP]\n')
        data = []

        for i in range(len(data_single)):
            if i+2 < len(data_single):
                temp = data_single[i].split('\n')
                n_temp = data_single[i+1].split('\n')
                n_n_temp = data_single[i+2].split('\n')
                
                temp[1] = temp[1].split('<|endoftext|>')[-2]
                n_temp[1] = n_temp[1].split('<|endoftext|>')[-2]
                n_n_temp[1] = n_n_temp[1].split('<|endoftext|>')[-2]
                # temp[2] = temp[2].replace('<|endoftext|>','')

                # Satisfy the coherent and consistent 0
                data.append([temp[0],temp[1],temp[2],0])
                
                # Don't meet 1
                data.append([temp[0],temp[1],n_temp[2],1])
                
                # Meet the same 2 
                if temp[0] == n_n_temp[0]:
                    data.append([temp[0],temp[1],n_n_temp[2],2])
                    
                # # Meet the coherent 3
                # data.append([temp[0],n_temp[1],n_temp[2],3])
                
                
        data_single = data
        
        self.data_single = data_single
        random.shuffle(self.data_single)
        if stage == 'fit' or stage is None:
            self.dataset_train = self.data_single[0:int(len(self.data_single)*0.9)]
            self.dataset_valid = self.data_single[int(len(self.data_single)*0.9):]
        if stage == 'test' or stage is None:
            self.dataset_valid = self.data_single[int(len(self.data_single)*0.9):]



    def collate_fn(self,batch):
        inputs_x = []
        inputs_y = []
        inputs_p = []
        inputs_c = []
        inputs_r = []
        for line in batch:
            inputs_x.append(('[CLS] ' + line[0] + ' [PERSONA] ' + line[1] + ' [CONTEXT] ' + line[2].replace('<|endoftext|>',' [RESPONSE]')))
            inputs_y.append(line[3])
            
            inputs_p.append('[CLS] ' + line[0] + ' [PERSONA]')
            inputs_c.append(line[1] + ' [CONTEXT]')
            inputs_r.append(line[2].replace('<|endoftext|>',' [RESPONSE]'))
        
        
        inputs_x = self.tok.batch_encode_plus(
            inputs_x,
            max_length=256,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
            )
        
        # print(inputs_x['input_ids'][0])
        # print(inputs_x['attention_mask'][0])
        # print(inputs_x['token_type_ids'][0])        
        
        l_num = 0
        for i in range(len(inputs_x['input_ids'])):
            temp_l = self.tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[PERSONA]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 0
            l_num = temp_l
            
            temp_l = self.tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[CONTEXT]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 1
            l_num = temp_l
            
            temp_l = self.tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[RESPONSE]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 2

        # print(inputs_x['token_type_ids'][0])     

        inputs = {'input_ids':inputs_x['input_ids'],'attention_mask':inputs_x['attention_mask'],'token_type_ids':inputs_x['token_type_ids']}
        labels = torch.tensor(inputs_y)
        
        return inputs,labels
        
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,num_workers=1,collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size,num_workers=1,collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size,num_workers=1,collate_fn=self.collate_fn)

    
# Classification model
class Class_Model(nn.Module):
    def __init__(self) -> None:
        super(Class_Model,self).__init__()
        
        self.sentence_embedding = nn.Embedding(3,768)
        config = Config()
        data_language = config.data_language        
        
        if data_language=='EN':
            self.model = RobertaForSequenceClassification.from_pretrained("xlm-roberta-base",num_labels=3)
        else:
            self.model = RobertaForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext",num_labels=3)

        self.sentence_embedding.weight.data[1:,:].copy_(self.model.roberta.embeddings.token_type_embeddings.weight.data)

    
    def forward(self,input_ids,attention_mask,token_type_ids,labels=None,return_dict=True):
        
        sentence_embeds = self.sentence_embedding(token_type_ids)
        inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        inputs_embeds = inputs_embeds + sentence_embeds
        
        
        outputs = self.model(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels = labels,
            return_dict=True)
        
        return outputs


class Consis_Model(pl.LightningModule):
    def __init__(self):
        super(Consis_Model,self).__init__()
        config = Config()
        data_language = config.data_language        
        
        if data_language=='EN':
            self.tok = RobertaTokenizer.from_pretrained("xlm-roberta-base",do_lower_case=False)
        else:
            self.tok = RobertaTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

        self.model = Class_Model()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,inputs,labels=None):
        outputs = self.model(input_ids=inputs['input_ids'],
            attention_mask= inputs['attention_mask'],
            token_type_ids = inputs['token_type_ids'],
            labels = labels,
            return_dict=True)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x,y)
        loss = outputs.loss
        self.log('train_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x,y)
        loss = outputs.loss
        # self.log('val_loss', loss)
        y_hat = outputs.logits
        return [y,y_hat,loss]

    def validation_epoch_end(self, outputs):
        # 计算平均损失
        for i in range(len(outputs)):
            if i == 0:
                y = outputs[i][0]
                y_hat = outputs[i][1]
                val_loss_sum = outputs[i][2]
            else:
                y = torch.cat([y,outputs[i][0]],dim=0)
                y_hat = torch.cat([y_hat,outputs[i][1]],dim=0)
                val_loss_sum += outputs[i][2]
        val_loss = val_loss_sum/len(outputs)
        self.log('val_loss',val_loss,on_epoch=True,prog_bar=True,logger=True)
        
        y_hat = torch.argmax(y_hat,dim=1)
        acc = torch.sum(y_hat == y).float() / y.shape[0]     
        self.log('val_acc',acc,on_epoch=True,prog_bar=True,logger=True)   

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).logits
        
        return [y,y_hat]

        
    def test_epoch_end(self, outputs):
        # 拼接
        for i in range(len(outputs)):
            if i == 0:
                y = outputs[i][0]
                y_hat = outputs[i][1]
            else:
                y = torch.cat([y,outputs[i][0]],dim=0)
                y_hat = torch.cat([y_hat,outputs[i][1]],dim=0)
                
        # 计算准确率
        y_hat = torch.argmax(y_hat,dim=1)
        acc = torch.sum(y_hat == y).float() / y.shape[0]
        
        self.log('test_acc',acc,on_epoch=True,prog_bar=True,logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),lr=1e-5,weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=8000, num_training_steps=84000)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        
        
        return [optimizer], [scheduler]
    
    def predict_dataloader(self):
        return super().predict_dataloader()

def collate_fn_predict(batch):
    config = Config()
    data_language = config.data_language        
    
    if data_language=='EN':
        tok = RobertaTokenizer.from_pretrained("xlm-roberta-base",do_lower_case=False)
    else:
        tok = RobertaTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    tok.sep_token = None
    # Add new words
    tok.add_tokens(['[PERSONA]','[CONTEXT]','[RESPONSE]'],special_tokens=True)
    inputs_x = []

    for line in batch:
        inputs_x.append(('[CLS] ' + line[0] + ' [PERSONA] ' + line[1] + ' [CONTEXT] ' + line[2] +' [RESPONSE]'))
        
    inputs_x = tok.batch_encode_plus(
        inputs_x,
        max_length=256,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=False,
        )
    
    l_num = 0
    for i in range(len(inputs_x['input_ids'])):
        temp_l = tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[PERSONA]')+1
        inputs_x['token_type_ids'][i,l_num:temp_l] = 0
        l_num = temp_l
        
        temp_l = tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[CONTEXT]')+1
        inputs_x['token_type_ids'][i,l_num:temp_l] = 1
        l_num = temp_l
        
        temp_l = tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[RESPONSE]')+1
        inputs_x['token_type_ids'][i,l_num:temp_l] = 2

    # print(inputs_x['token_type_ids'][0])     

    inputs = {'input_ids':inputs_x['input_ids'],'attention_mask':inputs_x['attention_mask'],'token_type_ids':inputs_x['token_type_ids']}
    
    labels = None
    
    return inputs,labels    
    
def predict(batch):
    config = Config()
    data_language = config.data_language   
    predict_dataloader = DataLoader(batch, batch_size=16,num_workers=0,collate_fn=collate_fn_predict)
    # Load the data
    model = Consis_Model()
    model = model.to(torch.device("cuda:0"))
    if data_language=='EN':
        d = torch.load(config.consis_model_dir_EN)
    else:
        d = torch.load(config.consis_model_dir_ZH)
    model.load_state_dict(d['state_dict'],strict=False)
    model.eval()
    
    y_predict = []
    for i, batch in enumerate(predict_dataloader):
        x, y = batch
        x = {k: v.to(torch.device("cuda:0")) for k, v in x.items()}
        y_hat = model(x).logits
        y_hat = torch.argmax(y_hat,dim=1)
        y_hat = y_hat.tolist()
        y_predict.extend(y_hat)
    
    return y_predict
        
    

if __name__=='__main__':
    pl.seed_everything(2022)
    model_file = 'Consis_Model'
    dm = Consis_Data()
    checkpoint_callback = ModelCheckpoint(
        dirpath='./Consis_model',
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only = True,
        filename= model_file + '-{epoch:02d}-{step}-{val_loss:.2f}-{val_acc:.2f}',
        save_last=True
    )
    learning_rate_callback = LearningRateMonitor()
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.0, 
        patience=3, 
        mode='min', 
        strict=True
    )
    logger = TensorBoardLogger('consis_log', name='test')
    
    dm.setup('fit')
    trainer = pl.Trainer(
        default_root_dir='./Consis_model',
        gradient_clip_val=0.5,
        max_epochs=10,
        gpus=1,
        val_check_interval=3000,
        callbacks=[learning_rate_callback, checkpoint_callback,early_stopping],
        logger = logger,
    )  
    model = Consis_Model()
    trainer.fit(model, datamodule=dm)
    
    dm.setup('test')
    
    trainer.test(model, datamodule=dm)
