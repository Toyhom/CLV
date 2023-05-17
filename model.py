import math
import torch
import torch.nn as nn
# from Embedding import Embedding
from Encoder import Encoder
from PriorNet import PriorNet
from RecognizeNet import RecognizeNet
from Decoder import Decoder
# from PrepareState import PrepareState
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytorch_lightning as pl


class Similarity(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class MLModel(pl.LightningModule):
    def __init__(self, config):
        super(MLModel, self).__init__()
        self.config = config

        # Encoder personality, reply, context are essentially text, so share a coder
        self.encoder = Encoder()  

        self.prior_net_p_list = []
        self.recognize_net_p_list = []

        for _ in range(config.N):
            self.prior_net_p_list.append(PriorNet(config.embedding_dim,
                                  config.latent_size,  
                                  config.dims_prior))
            self.recognize_net_p_list.append(RecognizeNet(config.embedding_dim,
                                          config.embedding_dim, 
                                          config.latent_size,
                                          config.dims_recognize))
        
        # self.cls_layer = nn.Sequential(nn.Linear(config.latent_size, config.latent_size),nn.Tanh)
        
        # 除以温度系数temp
        self.cos_sim = Similarity(temp=0.5)
        
        self.cls = nn.Linear(config.latent_size,config.latent_size)
        
        self.decider = nn.Sequential(nn.Linear(config.latent_size*config.N+config.embedding_dim,config.embedding_dim),
                                     nn.Tanh(),
                                     nn.Linear(config.embedding_dim,config.N+1))

        self.contra = nn.Sequential(nn.Linear(config.embedding_dim+config.embedding_dim,config.embedding_dim),
                                    nn.ReLU(),
                                    nn.Linear(config.embedding_dim,config.embedding_dim))

        # prior network-r
        self.prior_net = PriorNet(config.embedding_dim,  # The input dimension
                                  config.latent_size,  # Latent variable dimension
                                  config.dims_prior)  # Dimensions of hidden layers

        # recognition network-r
        self.recognize_net = RecognizeNet(config.embedding_dim, 
                                          config.embedding_dim,  
                                          config.latent_size,  
                                          config.dims_recognize)  


        # decoder
        self.decoder = Decoder()  

    def forward(self, inputs, is_train=False):
        text_posts = inputs['posts']  # [batch, seq]
        text_responses = inputs['responses']  # [batch, seq]
        text_posts_responses = inputs['p_r'] #only for train
        text_persons = inputs['persons']  # [batch, seq]
        sampled_latents = inputs['sampled_latents']  # [batch,latent_size]


        # state: [batch, seq ,dim]
        state_posts = self.encoder(text_posts)
        state_responses = self.encoder(text_responses)
        state_persons = self.encoder(text_persons)

        x = state_posts # [batch,dim]
        y = state_responses  # [batch,dim]
        p = state_persons  # [batch,dim]
        p_list = []
        # p_mean = torch.mean(p)
        N_length = self.config.embedding_dim//self.config.N
        for i in range(self.config.N):
            p_i = p.clone()
            
            c_t = torch.zeros_like(p_i)
            c_t = c_t.to(p_i.device)
            c_t[:,i*N_length:(i+1)*N_length] = 1
            
            p_i[:,i*N_length:(i+1)*N_length] += 1
            
            p_i = self.contra(torch.cat([p_i,c_t],dim=-1))
            
            p_i = p_i.unsqueeze(0)
            p_list.append(p_i)

        # [N,B,D]
        p_matrix = torch.stack(p_list).squeeze(1)
        # p_matrix = torch.cat(p_list,0).view(self.config.N,-1,self.config.embedding_dim)
        assert p_matrix.shape == torch.Size([self.config.N,sampled_latents.shape[0],self.config.embedding_dim])
        di_loss = torch.tensor(0.0).to(self.device)
        if not self.config.conditions['no_di'] or not self.config.conditions['no_bert']:
            if is_train:
                # Comparative study section
                # Each sample in the batch should be compared with other samples to study, reference simcse
                # Choose two samples from bacth, [N, 1, D] and [1, N, D] to calculate similarity matrix (N, N)
                # di_loss = torch.tensor(0.0).to(self.device)
                loss_fct = nn.CrossEntropyLoss()
                if sampled_latents.shape[0] >= 2:
                    # sample_res = torch.multinomial(torch.tensor([1.0/sampled_latents.shape[0]]*sampled_latents.shape[0]), 2, replacement=False)
                    # i = sample_res[0]
                    # j = sample_res[1]
                    # # [N,D]
                    # z1 = self.cls(p_matrix[:,i,:])
                    # z2 = self.cls(p_matrix[:,j,:])
                    # cos_sim = self.cos_sim(z1.unsqueeze(1),z2.unsqueeze(0))
                    # labels = torch.arange(cos_sim.size(0)).long().to(self.device)
                    # di_loss = loss_fct(cos_sim,labels)    
                                
                    for i in range(sampled_latents.shape[0]):
                        for j in range(i+1,sampled_latents.shape[0]):
                            # [N,D]
                            z1 = self.cls(p_matrix[:,i,:])
                            z2 = self.cls(p_matrix[:,j,:])
                            cos_sim = self.cos_sim(z1.unsqueeze(1),z2.unsqueeze(0))
                            labels = torch.arange(cos_sim.size(0)).long().to(self.device)
                            loss = loss_fct(cos_sim,labels)
                            di_loss+=loss
                    di_loss = di_loss/(sampled_latents.shape[0]*(sampled_latents.shape[0]-1)/2)
            p_matrix = self.cls(p_matrix)
        
        

        # Part of the personality

        _mu_p_list = []
        _logvar_p_list = []
        mu_p_list = []
        logvar_p_list = []
        z_p_list = []
        # [B,D]
        z_p_zero = torch.zeros((sampled_latents.shape[0],self.config.latent_size)).to(self.device)
        for i in range(self.config.N):
            _mu_p, _logvar_p = self.prior_net_p_list[i](x)
            mu_p, logvar_p = self.recognize_net_p_list[i](x, p_matrix[i])  # [batch, latent]
            if is_train:
                z_p = mu_p + (0.5 * logvar_p).exp() * sampled_latents  # [batch, latent]
            else:
                z_p = _mu_p + (0.5 * _logvar_p).exp() * sampled_latents  # [batch, latent]
            _mu_p_list.append(_mu_p)
            _logvar_p_list.append(_logvar_p)
            mu_p_list.append(mu_p)
            logvar_p_list.append(logvar_p)
            z_p_list.append(z_p)
        
        #(N+1)*[B,D] -> [N+1,B,D] => [B,N+1,D]
        z_p_matrix = torch.stack(z_p_list+[z_p_zero])
        z_p_matrix  = z_p_matrix.permute(1, 0, 2)
        assert z_p_matrix.shape == torch.Size([sampled_latents.shape[0],self.config.N+1,self.config.embedding_dim])
        # z_p_matrix = torch.cat(z_p_list+[z_p_zero],1).view(sampled_latents.shape[0],self.config.N+1,-1)
        #[B,1,N+1]
        decide_matrix = self.decider(torch.cat(z_p_list+[x],1)).unsqueeze(1)

        

        # softmax
        decide_matrix_softmax = torch.softmax(decide_matrix,dim=2)            
        # [B,D] = [B,1,N+1] * [B,N+1,D]
        z_p_w = torch.bmm(decide_matrix_softmax,z_p_matrix).squeeze(1)
        z_p = z_p_w
        
        
        # if self.config.hard_decision:
        #     # [B,1,D]
        #     hard_decide_matrix = torch.argmax(decide_matrix,2).unsqueeze(2).repeat(1,1,self.config.embedding_dim)
        #     # [B,D]
        #     z_p = z_p_matrix.gather(1,hard_decide_matrix).squeeze(1)        
        
        if self.config.conditions['no_decision']:
            z_p = torch.mean(z_p_matrix,1).squeeze(1)

        # response
        # p(z|q)  
        _mu, _logvar = self.prior_net(x)  # [batch, latent]
        # p(z|q,r)
        mu, logvar = self.recognize_net(x, y)  # [batch, latent]
        # parameterized
        if is_train:
            z_r = mu + (0.5 * logvar).exp() * sampled_latents  # [batch, latent]
        else:
            z_r = _mu + (0.5 * _logvar).exp() * sampled_latents
            
        # Pseudo tag, filter loss
        if is_train and self.config.conditions['false_label']:
            # No back propagation, save a space
            torch.no_grad()
            candidate_loss = []
            for i in range(z_p_matrix.shape[1]):
                candidate_loss_batch = []
                for j in range(z_p_matrix.shape[0]):
                    candidate_loss_batch.append(self.decoder(is_train,[text_persons[j]],[text_posts[j]],[text_responses[j]],[text_posts_responses[j]],z_r[j].unsqueeze(0),z_p_matrix[j,i,:].unsqueeze(0)))
                candidate_loss.append(torch.stack(candidate_loss_batch))
            
            # [N+1,B]
            candidate_loss = torch.stack(candidate_loss)
            # Loss to the minimum indexes
            loss_index = torch.argmin(candidate_loss,dim=0)
            
            # The cross entropy loss
            decide_loss_fct = nn.CrossEntropyLoss()
            
            decide_loss = decide_loss_fct(decide_matrix.squeeze(1),loss_index)
            

        decode_loss = self.decoder(is_train,text_persons,text_posts,text_responses,text_posts_responses,z_r,z_p)
        if is_train and self.config.conditions['false_label']:
            decode_loss = decide_loss + decode_loss
        if is_train:
            return _mu, _logvar, mu, logvar,_mu_p_list, _logvar_p_list, mu_p_list, logvar_p_list,decode_loss,di_loss
        else:
            return decode_loss



    def print_parameters(self):
        r""" Statistical parameter """
        total_num = 0  
        for param in self.parameters():
            num = 1
            if param.requires_grad:
                size = param.size()
                for dim in size:
                    num *= dim
            total_num += num
        print(f"Total number of parameters: {total_num}")




