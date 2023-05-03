from rouge import Rouge
from tqdm import tqdm 
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk import word_tokenize, sent_tokenize 
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk import bigrams, FreqDist
import torch
from Consis_Model import predict
from config import Config

# NLP generated in the task of evaluation
class NLP_Gen_Matrics:
    def __init__(self,condition_list=['test'],model_name='test_model'):
        super(NLP_Gen_Matrics, self).__init__()
        self.judge_value_list = dict()
        self.condition_list = condition_list
        self.model_name = model_name
        self.config = Config()
    
    def get_judge_data(self,result_path):
        # format: '我[CSE]是[CSE]你\n我[CSE]是[CSE]他\n\n'
        with open(result_path,'r',encoding='utf-8') as f:
            data = f.read()
        data = data.split('\n\n')
        persona_list = []
        context_list = []
        pred_list = []
        target_list = []
        for i in range(len(data)):
            temp = data[i]
            config = Config()
            data_language = config.data_language
            
            if data_language=='EN':
                # Punctuation handling
                char_list = [',','.','!','?','，','。','！','？']
                for c in char_list:
                    temp = temp.replace('[CSE]' + c,'[CSE]Ġ'+c)
        
                
                # All converted to lowercase letters
                temp = temp.replace('[CSE]Ġ',' ')
                temp = temp.replace('[CSE]','').replace('\t','')

                temp = temp.lower()
                temp = temp.split('\n')
            else:
                temp = temp.replace('\t','').replace('[CSE]',' ')
                temp = temp.lower()
                temp = temp.split('\n')
                
            
            if len(temp)==4:
                # Remove all empty string in the list
                persona_list.append(list(filter(None, temp[0].split(' '))))
                context_list.append(list(filter(None, temp[1].split(' '))))
                
                temp_pred_list = []
                for j in temp[2].split('###'):
                    temp_pred_list.append(list(filter(None, j.split(' '))))
        
                pred_list.append(temp_pred_list)
                
                target_list.append(list(filter(None, temp[3].split(' '))))
        
        print(pred_list[0],target_list[0])
        
        with open(result_path.replace('.txt','_process.txt'),'w',encoding='utf-8') as f:
            f.write('\n\n'.join([' '.join(persona_list[i]) + '\n' + ' '.join(context_list[i]) + '\n' + '###'.join([' '.join(pred_list[i][j]) for j in range(len(pred_list[i]))]) + '\n' + ' '.join(target_list[i]) for i in range(len(pred_list))]))
        
        return self.get_result(persona_list,context_list,pred_list,target_list)
        
        
    def get_result(self,persona_list,context_list,pred_list,target_list):
        self.get_distinct(pred_list,target_list)
        self.get_coherence(pred_list,target_list)
        self.get_consistency(persona_list,context_list,pred_list,target_list)
        
        
        file_name = '_'.join(self.condition_list) + '_' + self.model_name
        # Save the result
        with open('./result/' + file_name + '-metrics.txt','w',encoding='utf-8') as f:
            f.write('condition_list:  '+ self.model_name + str(self.condition_list) + '\n')
            for key,value in self.judge_value_list.items():
                f.write(key + ' : ' + str(value) + '\n')
        
        return self.judge_value_list
            
    
    # Local diversity Distinct - 1, 2
    def get_distinct(self, pred,target):
        
        temp_s_dist_1 = []
        temp_s_dist_2 = []
        for m_pred in pred:
            unigrams = []
            bigrams = []
            for n,rep in enumerate(m_pred):
                temp = rep
                unigrams += temp
                for i in range(len(temp)-1):
                    bigrams.append(temp[i] + ' ' + temp[i+1])
            if len(unigrams) != 0:
                temp_s_dist_1.append(len(set(unigrams)) / len(unigrams))
            if len(bigrams) != 0:
                temp_s_dist_2.append(len(set(bigrams)) / len(bigrams))

        self.judge_value_list['s-distinct-1'] = sum(temp_s_dist_1) / len(temp_s_dist_1)
        self.judge_value_list['s-distinct-2'] = sum(temp_s_dist_2) / len(temp_s_dist_2)
        
        corpus = [i[0] for i in pred]
        unigrams = []
        bigrams = []
        for n,rep in enumerate(corpus):
            temp = rep
            unigrams += temp
            for i in range(len(temp)-1):
                bigrams.append(temp[i] + ' ' + temp[i+1])
        self.judge_value_list['c-distinct-1'] = len(set(unigrams)) * 1.0 / len(unigrams)
        self.judge_value_list['c-distinct-2'] = len(set(bigrams)) * 1.0 / len(bigrams)
        

    # Global diversity
    def get_diversity(self):
        pass
    
    # consistency
    def get_consistency(self,persona_list,context_list,pred_list,target_list):
        if self.config.model_metric == True:
            pred_list = [i[0] for i in pred_list]
            batch_data = []
            for i in range(len(pred_list)):
                batch_data.append([' '.join(persona_list[i]),' '.join(context_list[i]),' '.join(pred_list[i])])
            y_hat = predict(batch_data)
            # Calculate according to the category scores
            # y_hat In the number of category 2

            self.judge_value_list['consistency-coherence'] = y_hat.count(2) / len(y_hat)
            self.judge_value_list['consistency'] = (y_hat.count(1) + y_hat.count(2)) / len(y_hat)
    
    
    # coherence
    def get_coherence(self,pred_list,target_list):
        
        # bleu
        num = 0
        bleu_score_all_1 = 0
        bleu_score_all_2 = 0
        bleu_score_all_3 = 0
        bleu_score_all_4 = 0
        for pred,target in zip(pred_list,target_list):
            bleu_score_1 = sentence_bleu(target, pred[0],weights=(1, 0, 0, 0))
            bleu_score_all_1 += bleu_score_1
            bleu_score_2 = sentence_bleu(target, pred[0],weights=(0.5, 0.5, 0, 0))
            bleu_score_all_2 += bleu_score_2
            bleu_score_3 = sentence_bleu(target, pred[0],weights=(0.33, 0.33, 0.33, 0))
            bleu_score_all_3 += bleu_score_3
            bleu_score_4 = sentence_bleu(target, pred[0],weights=(0.25, 0.25, 0.25, 0.25))
            bleu_score_all_4 += bleu_score_4
            num+=1
        self.judge_value_list['bleu-1'] = bleu_score_all_1/num
        self.judge_value_list['bleu-2'] = bleu_score_all_2/num
        self.judge_value_list['bleu-3'] = bleu_score_all_3/num
        self.judge_value_list['bleu-4'] = bleu_score_all_4/num        
        
        pred_list = [i[0] for i in pred_list]
        
        # rouge
        rouge = Rouge()
        rouge_list = [[],[],[]]
        for pred,target in zip(pred_list,target_list):
            if len(target) <= 1:
                continue 
            if len(pred) <= 1:
                pred.append('<UNK>')
            if len(target) <= 1:
                target.append('<UNK>')
            rouge_score = rouge.get_scores(" ".join(pred), " ".join(target))
            rouge_list[0].append(rouge_score[0]['rouge-1']['r'])
            rouge_list[1].append(rouge_score[0]['rouge-2']['r'])
            rouge_list[2].append(rouge_score[0]['rouge-l']['r'])
        self.judge_value_list['rouge-1'] = sum(rouge_list[0]) / len(rouge_list[0])
        self.judge_value_list['rouge-2'] = sum(rouge_list[1]) / len(rouge_list[1])
        self.judge_value_list['rouge-l'] = sum(rouge_list[2]) / len(rouge_list[2])

    
# A filter, choose the best response in more generated reply
def get_best(pred_list,target):
    # bleu
    target_temp = [i.replace('Ġ','') for i in target]
    bleu_score_2_list = []
    for pred in pred_list:
        pred_temp = [i.replace('Ġ','') for i in pred]
        bleu_score_2 = sentence_bleu(target_temp,pred_temp,weights=(0.5, 0.5, 0, 0))
        bleu_score_2_list.append(bleu_score_2)
    
    # Select the optimal response of the serial number
    index = torch.argmax(torch.tensor(bleu_score_2_list))
    
    return pred_list[index]
        

if __name__ == '__main__':

    model_file = 'MLVGen-dropout-mask-N=4_ZH_false_label'
    judger = NLP_Gen_Matrics(model_name=model_file,condition_list=['2023-01-07'])
    result = judger.get_judge_data('./result/' + model_file + '-pred_result.txt')

    for key in result.keys():
        print(key,result[key])
        
