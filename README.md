# DeepMutation

It's a repository, which aims to implement mutation operators for Deep Learning Mutation Testing.  
  
The objective is to provide a tool to generate mutated models for mutation testing on Deep Learning system, where mutated models will be produced by mutation operators. In this repository, 8 source-level mutation operators and 8 model level mutation operators will be implemented.  
  
The concept of these mutation operators is introduced in the paper, <b> DeepMutation: Mutation Testing of Deep Learning Systems </b>, where the link is attached in references. However,  coding implementations of each mutation operators are not explicitly explained. In this repository, we clarify the vague part and document how each mutation operators be implemented, aiming to present a convenient tool for mutation testing on Deep Learning system.  


Source-level mutation operators 
------------------
Source-level mutation operators mutate either the original training dataset or the original training program. A training dataset mutant or training program mutant participates in the training process to generate a mutated model, donated as M'.  
  
For each of the mutation operators, it should be capable to generate several mutated models based on the same original training dataset and training program. Therefore, randomness should be involved in each of the mutation operators. See the description of individual operators Implementation for more details.   
  
-  <b> DR - Data Repetition:</b>  
   Target : Data  
   Brief Operator Description: DR operator duplicates a small portion of training data according to mutation ratio.
   Implementation:  
   1. Randomly select a specific amount of samples based on the mutation ratio, where each of the selected samples is chosen independently and exclusively. For instance, if there are 5000 samples and mutation ratio is set to be 0.01, 50 samples will be selected for duplication, where the samples should like [sample_3827, sample_2, sample_4999, ..., sample 2387] instead of [sample_1, sample_2, ..., sample_50] or [sample_4951, sample_4952, ..., sample_5000].  
   2. Selected samples are concatenated with the original training dataset.  
   
-  LE - Label Error 

-  DM - Data Missing 

-  DF - Data Shuffle
-  NP - Noise Perturb
-  LR - Layer Removal
-  LAs - Layer Addition (source-level)
-  AFRs - Activation Function Removal (source-level)


Background
----------------
  (some background about mutation operators will be added here)
  
 
Configuration
----------------
  Python: 3.5.1 <br/>
  Tensorflow: 1.12.0 <br/>
  NumPy: 1.15.1 <br/>


Installation
------------

    Currently not available
    


References
----------------
  Lei Ma, Fuyuan Zhang, Jiyuan Sun, Minhui Xue, Bo Li, Felix Juefei-Xu, Chao Xie, Li Li, Yang Liu, Jianjun Zhao, et al. <br/>
  DeepMutation:  Mutation testing of Deep Learning Systems. <br/>
  https://arxiv.org/pdf/1805.05206.pdf <br/>

  Jingyi Wang, Guoliang Dongy, Jun Sun, Xinyu Wangy, Peixin Zhangy, Singapore University of Technology and Design Zhejiang University.<br/> 
  Adversarial Sample Detection for Deep Neural Network through Model Mutation Testing <br/>
  https://arxiv.org/pdf/1812.05793.pdf <br/>
