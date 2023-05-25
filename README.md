# CLV

This repository provides the implementation details for the ACL 2023 main conference paper:

**Enhancing Personalized Dialogue Generation with Contrastive Latent Variables: Combining Sparse and Dense Persona**


## 0.Preinstallation
```python
  git clone CLV
  cd CLV
  pip install -r requirements.txt
```

## 1.Parts that may need to be modified
In ```config.py```:
```python
    # mdoel_file
    output_dir = './Model'

    # evaluation_model_file
    consis_model_dir_EN = './Consis_Model/consis_model_EN.ckpt'
    consis_model_dir_ZH = './Consis_Model/consis_model_ZH.ckpt'
    
    # Data file format
    data_path = ['/Data/Persona_Dialoue_EN','/Data/Persona_Dialoue_ZH']
    
    # Whether to use learnable metrics, 
    # and if so, train the Consis_Model first
    model_metric = False

    # language switch
    data_language = 'EN' 

```


## 2.Data file format
- See the samples in the **Data** folder.

- Sample:
```text
i have amazing children and grandchildren.i can sew my own clothes.i had cancer but its gone now.i am retired and living the great life.i do not have a smartphone.
<|endoftext|>I love iphone! i just bought new iphone!<|endoftext|>Thats good for you, i'm not very into new tech<|endoftext|>
I am a college student and i am a college student<|endoftext|>
```



## 3.Running
- Setiing ```config.py```,
- Then run ```clv.py```.
- It will be tested and evaluated automatically.



## 4.Training Consis_Model(if you need)
- Running ```Consis_Model.py```;
- Modify the model name according to **consis_model_dir_EN** or **consis_model_dir_ZH**;
- Execute **3.Running**.


-------
## MISC
* Build upon 🤗 [Transformers](https://github.com/huggingface/transformers).




