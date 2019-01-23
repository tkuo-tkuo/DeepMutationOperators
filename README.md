# DeepMutation

It's a repository, which aims to implement mutation operators for Deep Learning Mutation Testing.  
  
The objective is to provide a tool to generate mutated models for mutation testing on Deep Learning system, where mutated models will be produced by mutation operators. In this repository, 8 source-level mutation operators and 8 model level mutation operators will be implemented.  
  
The concept of these mutation operators is introduced in the paper, <b> DeepMutation: Mutation Testing of Deep Learning Systems </b>, where the link is attached in references. However,  coding implementations of each mutation operators are not explicitly explained. In this repository, we clarify the vague part and document how each mutation operators be implemented, aiming to present a convenient tool for mutation testing on Deep Learning system.  


Source-level mutation operators 
------------------
Source-level mutation operators mutate either the original training dataset or the original training program. A training dataset mutant or training program mutant participates in the training process to generate a mutated model, donated as M'.  
  
For each of the mutation operators, it should be capable to generate several mutated models based on the same original training dataset and training program. Therefore, randomness should be involved in each of the mutation operators. See the description of individual operators Implementation for more details.   
  
-  <b> DR - Data Repetition:</b>  
   Target : Training dataset  
   Brief Operator Description: DR operator duplicates a portion of training dataset.
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively based on mutation ratio. For instance, if there are 5000 samples and mutation ratio is set to be 0.01, 50 samples will be selected for duplication, where the samples should like [sample_3827, sample_2, sample_4999, ..., sample 2387] instead of [sample_1, sample_2, ..., sample_50] or [sample_4951, sample_4952, ..., sample_5000].  
   2. Selected samples are concatenated with the original training dataset.  
   
-  <b>LE - Label Error:</b>  
   Target: Training dataset  
   Brief Operator Description: LE operator falsifies a portion of results (e.g., labels) in traning dataset.  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Each result (e.g., label) among the chosen samples is mislabeled by LE operator. For instance, if the set of labels is donated as L, {0, 1, ..., 9}, and the correct label is 0, LE operator will randomly assign a value among L except the correct label 0.  
    
-  <b>DM - Data Missing :</b>  
   Target: Training dataset  
   Brief Operator Description: Remove a portion of training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively for further removal based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Selected samples in the training dataset are removed.  
   
-  <b>DF - Data Shuffle:</b>   
   Target: Training dataset  
   Brief Operator Description: Shuffle selected samples among training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusivel based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Only the selected samples will be shuffled and the order of unselected samples is preserved.  
   
-  <b>NP - Noise Perturb:</b>  
   Target: Training dataset  
   Brief Operator Description: Add noise to a portion of training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusivel based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Noises are appended on each of the selected datasets. Since raw data in the training dataset are rescaled in the range between 0 and 1, the value of noises follows normal distribution N(0, 0.1^2), where standard deviation parameter is a user-configurable parameter with default value 0.1.    
   
-  <b>LR - Layer Removal:</b>  
   Target: Training program  
   Brief Operator Description: Remove a layer   
   Implementation:  
   1. LR operator randomly deletes a layer on the condition that the input and output structure of the deleted layer are the same. It traverses the entire structure of Deep Learning model and records all the layers where the condition is satisfied.  
   2. According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, LR mutation operator mainly focuses on layers (e.g., Dense, BatchNormalization layer), whose deletion doesn't make too much influence on the mutated model, since arbitrary removal of a layer may generate obviously different DL model from the original one.  
   3. One of the selected layers which are recorded in step i. and satisfies the requirement of step ii. is randomly removed  
  
   Remarks that in my implementation, the input and output will not be included in the consideration.   
   
   
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
