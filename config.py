
class Config(object):
    r""" Model Configuration. """
    output_dir = './Model'
    data_path = ['/Data/Persona_Dialoue_EN','/Data/Persona_Dialoue_ZH']
    model_metric = False
    consis_model_dir_EN = './Consis_Model/consis_model_EN.ckpt'
    consis_model_dir_ZH = './Consis_Model/consis_model_ZH.ckpt'


    # ---------Data phase
    
    # Data language
    data_language = 'EN'    
    
    model_file = 'None'
    
    max_len = 128
    
    # --------Model construction phase
    
    # self-separation factor
    N = 4
    
    # Ablation conditions
    conditions = {'no_di':False,'no_bert':False,'no_decision':False,'no_kl':False,'false_label':True}
    
    # Decision-making way
    hard_decision = False
    
    # Word vector dimension
    embedding_dim = 768
    
    # prior network parameters
    dims_prior = [250]  # Multilayer perceptron prior network layer on the number of hidden units, like [the dim1 and dim2,..., dimn] such incoming
    # recognition network parameters
    dims_recognize = [250]  
    
    latent_size = embedding_dim
    latent_dim = embedding_dim
    post_encoder_output_size = embedding_dim
    response_encoder_output_size = embedding_dim
        
    # ---------------Operation phase
    
    
    # ---------------Inference phase
    # Generate sentences
    generate_max_len = 32  
    repetition_penalty = 1
    topk = 0
    topp = 0.9


    # The temperature of the similarity
    temp = 0.5
    



    # To optimize the parameters
    batch_size = 8
    epochs = 5
    lr = 1e-4  # The initial vector
    seed = 2022



    dims_prior = [128,128]
    dims_recognize = [128,128]

    print_per_step = 500//batch_size

    result_path =  './result'