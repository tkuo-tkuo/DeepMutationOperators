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
   
   Input: training dataset, training model, and mutation ratio    
   Output: mutated training dataset by DR operator and copied training model  
   Syntax:  
   ```js
    mutated_dataset, copied_model  = source_mut_opts.DR_mut(training_dataset, model, mutation_ratio)
   ```
   Example:  
   ```js
    (DR_train_datas, DR_train_labels), DR_model = source_mut_opts.DR_mut((train_datas, train_labels), model, 0.01)
   ```
   
-  <b>LE - Label Error:</b>  
   Target: Training dataset  
   Brief Operator Description: LE operator falsifies a portion of results (e.g., labels) in traning dataset.  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Each result (e.g., label) among the chosen samples is mislabeled by LE operator. For instance, if the set of labels is donated as L, {0, 1, ..., 9}, and the correct label is 0, LE operator will randomly assign a value among L except the correct label 0.    
   
   Input: training dataset, training model, label lower bound, label upper bound, and mutation ratio    
   Output: mutated training dataset by LE operator and copied training model  
   Syntax:  
   ```js
    mutated_dataset, copied_model  = source_mut_opts.LE_mut(training_dataset, model, label_lower_bound, label_upper_bound, mutation_ratio)
   ```
   Example:  
   ```js
    (LE_train_datas, LE_train_labels), LE_model = source_mut_opts.LE_mut((train_datas, train_labels), model, 0, 9, 0.01)
   ```
   
    
-  <b>DM - Data Missing :</b>  
   Target: Training dataset  
   Brief Operator Description: Remove a portion of training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively for further removal based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Selected samples in the training dataset are removed.  
      
   Input: training dataset, training model, and mutation ratio    
   Output: mutated training dataset by DM operator and copied training model  
   Syntax:  
   ```js
    mutated_dataset, copied_model  = source_mut_opts.DM_mut(training_dataset, model, mutation_ratio)
   ```
   Example:  
   ```js
    (DM_train_datas, DM_train_labels), DM_model = source_mut_opts.DM_mut((train_datas, train_labels), model, 0.01)
   ```
   
-  <b>DF - Data Shuffle:</b>   
   Target: Training dataset  
   Brief Operator Description: Shuffle selected samples among training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusivel based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Only the selected samples will be shuffled and the order of unselected samples is preserved.  
   
   Input: training dataset, training model, and mutation ratio    
   Output: mutated training dataset by DF operator and copied training model  
   Syntax:  
   ```js
    mutated_dataset, copied_model  = source_mut_opts.DF_mut(training_dataset, model, mutation_ratio)
   ```
   Example:  
   ```js
    (DF_train_datas, DF_train_labels), DF_model = source_mut_opts.DF_mut((train_datas, train_labels), model, 0.01)
   ```
   
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
   2. According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, LR mutation operator mainly focuses on layers (e.g., Dense, BatchNormalization layer), whose deletion doesn't make too much influence on the mutated model, since arbitrary removal of a layer may generate obviously different Deep Learning model from the original one.  
   3. One of the selected layers which are recorded in step i. and satisfies the requirement of step ii. is randomly removed  
  
   Remarks that in my implementation, the input layer and output layer will not be included in the consideration.   
   
-  <b>LAs - Layer Addition (source-level):</b>  
   Target: Training program  
   Brief Operator Description: Add a layer   
   Implementation:  
   1. According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, LAs operator mainly focuses on adding layers like Activation, BatchNormalization. More types of layers should be considered in the future implementation once addition of a layer will not generate obviously different Deep Learning model from the original one, where unqualified mutant can be filtered out.   

-  <b>AFRs - Activation Function Removal (source-level):</b>  
   Target: Training program  
   Brief Operator Description: Remove activation layers of a layer    
   Implementation:  
   1. AFRs randomly remove all activation functions of a layer.  
  
   Remarks that in my implementation, the activation functions of the output layer will not be included in the consideration. For instance, the value after activation function, softmax, of the output layer reflects the level of confidence. It may be better not to eliminate the activation functions of the output layer.  
   
Model-level mutation operators 
------------------
Model-level mutation operators directly mutate the structure and parameters of DNN's structure without training procedure, which is more efficient for mutated model generation. Explicitly, model-level mutation operators automatically analysis structure of given DNN and mutate on a copy of the original DNN, where the generated mutant models are serialized and stored as .h5 file format.  
  
-  <b>GF - Gaussian Fuzzing:</b>  
   Target: Trained model (Weight)  
   Brief Operator Description: Fuzz weight by Gaussian Distribution  
   Implementation:  
   1.   
   
-  <b>WS - Weight Shuffling:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Shuffle weights of selected neurons    
   Implementation:  
   1.   
   
   
-  <b>NEB - Neuron Effect Block:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Block effect of selected neurons on following layers    
   Implementation:  
   1.   
   
-  <b>NAI - Neuron Activation Inverse:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Invert (the sign) of activation status of selected neurons    
   Implementation:  
   1.   
   
-  <b>NS - Neuron Switch:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Switch neurons among the same layer    
   Implementation:  
   1.   
   
-  <b>LD - Layer Deactivation:</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Deactivate the effects of a layer    
   Implementation:  
   1.   
   
-  <b>LAm - Layer Addition (model-level):</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Add (copy) a layer in neural network    
   Implementation:  
   1.   
   
-  <b>AFRm - Activation Function Removal (model-level):</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Remove activation functions of a layer    
   Implementation:  
   1.   


Background
----------------
  (some background information about mutation operators will be added here)
  
 
Configuration
----------------
  Python: 3.5.1  
  Tensorflow: 1.11.0  
  Keras: 2.2.4  
  NumPy: 1.15.1  


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
