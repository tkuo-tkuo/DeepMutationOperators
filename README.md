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
   Output: mutated training dataset and copied training model  
   Syntax:  
   ```python
    mutated_dataset, copied_model  = source_mut_opts.DR_mut(training_dataset, model, mutation_ratio)
   ```
   Example:  
   ```python
    (DR_train_datas, DR_train_labels), DR_model = source_mut_opts.DR_mut((train_datas, train_labels), model, 0.01)
   ```
   
-  <b>LE - Label Error:</b>  
   Target: Training dataset  
   Brief Operator Description: LE operator falsifies a portion of results (e.g., labels) in traning dataset.  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Each result (e.g., label) among the chosen samples is mislabeled by LE operator. For instance, if the set of labels is donated as L, {0, 1, ..., 9}, and the correct label is 0, LE operator will randomly assign a value among L except the correct label 0.    
   
   Input: training dataset, training model, label lower bound, label upper bound, and mutation ratio    
   Output: mutated training dataset and copied training model  
   Syntax:  
   ```python
    mutated_dataset, copied_model  = source_mut_opts.LE_mut(training_dataset, model, label_lower_bound, label_upper_bound, mutation_ratio)
   ```
   Example:  
   ```python
    (LE_train_datas, LE_train_labels), LE_model = source_mut_opts.LE_mut((train_datas, train_labels), model, 0, 9, 0.01)
   ```
   
    
-  <b>DM - Data Missing :</b>  
   Target: Training dataset  
   Brief Operator Description: Remove a portion of training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively for further removal based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Selected samples in the training dataset are removed.  
      
   Input: training dataset, training model, and mutation ratio    
   Output: mutated training dataset and copied training model  
   Syntax:  
   ```python
    mutated_dataset, copied_model  = source_mut_opts.DM_mut(training_dataset, model, mutation_ratio)
   ```
   Example:  
   ```python
    (DM_train_datas, DM_train_labels), DM_model = source_mut_opts.DM_mut((train_datas, train_labels), model, 0.01)
   ```
   
-  <b>DF - Data Shuffle:</b>   
   Target: Training dataset  
   Brief Operator Description: Shuffle selected samples among training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusivel based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Only the selected samples will be shuffled and the order of unselected samples is preserved.  
   
   Input: training dataset, training model, and mutation ratio    
   Output: mutated training dataset and copied training model  
   Syntax:  
   ```python
    mutated_dataset, copied_model  = source_mut_opts.DF_mut(training_dataset, model, mutation_ratio)
   ```
   Example:  
   ```python
    (DF_train_datas, DF_train_labels), DF_model = source_mut_opts.DF_mut((train_datas, train_labels), model, 0.01)
   ```
   
-  <b>NP - Noise Perturb:</b>  
   Target: Training dataset  
   Brief Operator Description: Add noise to a portion of training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusivel based on mutation ratio. See the illustration in DR Implementation step i.  
   2. Noises are appended on each of the selected datasets. Since raw data in the training dataset are rescaled in the range between 0 and 1, the value of noises follows normal distribution, where standard deviation parameter is a user-configurable parameter with default value 0.1 and mean is 0.    
      
   Input: training dataset, training model, and mutation ratio    
   Output: mutated training dataset and copied training model  
   Syntax:  
   ```python
    mutated_dataset, copied_model  = source_mut_opts.NP_mut(training_dataset, model, mutation_ratio, STD=0.1)
   ```
   Example:  
   ```python
    # Without specification of standard deviation parameter (STD), STD is set to 0.1 as default
    (NP_train_datas, NP_train_labels), NP_model = source_mut_opts.NP_mut((train_datas, train_labels), model, 0.01)
    # Usage with specification of STD value as 2
    (NP_train_datas, NP_train_labels), NP_model = source_mut_opts.NP_mut((train_datas, train_labels), model, 0.01, STD=2)
   ```
   
-  <b>LR - Layer Removal:</b>  
   Target: Training program  
   Brief Operator Description: Remove a randomly selected layer which satisfies conditions  
   Implementation:  
   1. LR operator traverses through the entire structure of deep learning model and record all the layers where conditions are satisfied. The first condition is that the input and output shape of a layer should be the same. The second condition is that, according to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, LR mutation operator mainly focuses on layers (e.g., Dense, BatchNormalization layer), whose deletion doesn't make too much influence on the mutated model, since arbitrary removal of a layer may generate obviously different Deep Learning model from the original one.  
   2. One of the selected layers is randomly removed from the deep learning model.  
   
   Input: training dataset and training model  
   Output: copied training dataset and mutated training model  
   Syntax:  
   ```python
    copied_dataset, mutated_model  = source_mut_opts.LR_mut(training_dataset, model)
   ```
   Example:  
   ```python
    (LR_train_datas, LR_train_labels), LR_model = source_mut_opts.LR_mut((train_datas, train_labels), model)
   ```
  
   Remarks that in my implementation, the input layer and output layer will not be included in the consideration.   
   
-  <b>LAs - Layer Addition for source-level:</b>  
   Target: Training program  
   Brief Operator Description: Randomly add a layer to one of suitable spots in the deep learning model  
   Implementation:  
   1. LAs operator traverses through the entire structure of deep learning model and record all the spots where a layer can be added.   
   2. According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, LAs operator mainly focuses on adding layers like Activation, BatchNormalization. More types of layers should be considered in the future implementation once addition of a layer will not generate obviously different Deep Learning model from the original one, where unqualified mutant can be filtered out.   
  
   Input: training dataset and training model  
   Output: copied training dataset and mutated training model  
   Syntax:  
   ```python
    copied_dataset, mutated_model  = source_mut_opts.LAs_mut(training_dataset, model)
   ```
   Example:  
   ```python
    (LAs_train_datas, LAs_train_labels), LAs_model = source_mut_opts.LAs_mut((train_datas, train_labels), model)
   ```
  
   Remarks that in my implementation, the output layer will not be included in the consideration.   

-  <b>AFRs - Activation Function Removal for source-level:</b>  
   Target: Training program  
   Brief Operator Description: Remove activation layers of a randomly selected layer    
   Implementation:  
   1. AFRs operator traverses through the entire structure of deep learning model and record all the layers with activation functions except the output layer.  
   1. AFRs randomly remove all activation functions of a randomly selected layer.  
     
   Input: training dataset and training model  
   Output: copied training dataset and mutated training model  
   Syntax:  
   ```python
    copied_dataset, mutated_model  = source_mut_opts.AFRs_mut(training_dataset, model)
   ```
   Example:  
   ```python
    (AFRs_train_datas, AFRs_train_labels), AFRs_model = source_mut_opts.AFRs_mut((train_datas, train_labels), model)
   ```
   
   Remarks that in my implementation, the activation functions of the output layer will not be included in the consideration. For instance, the value after activation function, softmax, of the output layer reflects the level of confidence. It may be better not to eliminate the activation functions of the output layer.  
   
Model-level mutation operators 
------------------
Model-level mutation operators directly mutate the structure and parameters of DNN's structure without training procedure, which is more efficient for mutated model generation. Explicitly, model-level mutation operators automatically analysis structure of given DNN and mutate on a copy of the original DNN, where the generated mutant models are serialized and stored as .h5 file format.  
  
-  <b>GF - Gaussian Fuzzing:</b>  
   Target: Trained model (Weight)  
   Brief Operator Description: Fuzz a portion of weights in trained model by Gaussian Distribution   
   Implementation:  
   1. For weights of each layer, GF flattens weights of each layer to a one-dimensional list, since GF does not need to recognize the relationship between neurons. A one-dimensional list will be handy for manipulation.  
	 2.  GF mutation operator chooses elements among the one-dimensional list independently and exclusively based on mutation ratio.  
	 3. GF mutation operators add noise on selected weight, where the noise follows normal distribution ~N(0, std^2). The standard deviation parameter is user-configurable with default value as 0.01.  
   
   Input: trained model, mutation ratio, and standard deviation   
   Output: mutated trained model   
   Syntax:  
   ```python
    mutated_model  = model_mut_opts.GF_mut(model, mutation_ratio, STD=0.1)
   ```
   Example:  
   ```python
    # Without specification of standard deviation parameter (STD), STD is set to 0.1 as default
    GF_model = model_mut_opts.GF_mut(model, 0.01)
    # Usage with specification of STD value as 2
    GF_model = model_mut_opts.GF_mut(model, 0.01, STD=2)
   ```
   
   Remarks that GF mutation operator works well with Dense, Activation, batch normalization. However, it's not guaranteed for convolutional layer yet. It's still under development.  
   
-  <b>WS - Weight Shuffling:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Shuffle weights to which selected neurons connect to previous layer   
   Implementation:  
   1.  Except for the input layer, for each layer, select neurons independently and exclusively based on the mutation ratio 
   2.  Shuffle the weights of each neuron's connections to the previous layer. For instance, the weights of Dense layer are stored in a matrix (2-dimension list) m * n. If neuron j is selected, all the weights w[:, j] connecting to neuron j are extracted, shuffled, and injected back in a matrix.  
   
   Input: trained model and mutation ratio  
   Output: mutated trained model   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.WS_mut(model, mutation_ratio)
   ```
   Example:  
   ```python
   WS_model = model_mut_opts.WS_mut(model, 0.01)
   ```
   
   Remarks that biases are excluded for consideration and WS mutation operator for convolutional layer is still under development.  
     
-  <b>NEB - Neuron Effect Block:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Block effect of selected neurons on following layers    
   Implementation:  
   1. Except for the output layer, for each layer, select neurons independently and exclusively based on the mutation ratio.  
   2. Block the effect of selected neurons by setting all the weights connecting to next layer as 0.  
   
   Input: trained model and mutation ratio  
   Output: mutated trained model   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.NEB_mut(model, mutation_ratio)
   ```
   Example:  
   ```python
   NEB_model = model_mut_opts.NEB_mut(model, 0.01)
   ```
   
   Remarks that biases are excluded for consideration.
   
-  <b>NAI - Neuron Activation Inverse:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Invert (the sign) of activation status of selected neurons    
   Implementation:  
   1. Except for the input layer, for each layer, select neurons independently and exclusively based on the mutation ratio.  
   2. According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, invertion of the activation status of a neuron can be achieved by changing the sign of a neuron's value before applying its activation function. This can be actually achieved by multiplying -1 to all the weights connecting on the previous layer of selected neurons since the output value of a neuron before applying its activation function is the sum of product connecting to a neuron.  
    
   Input: trained model and mutation ratio  
   Output: mutated trained model   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.NAI_mut(model, mutation_ratio)
   ```
   Example:  
   ```python
   NAI_model = model_mut_opts.NAI_mut(model, 0.01)
   ```
   
   Remarks that NAI mutation operator for convolutional layer is still under development.  
   
-  <b>NS - Neuron Switch:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Switch two neurons (shuffle neurons) of the same layer  
   Implementation:  
   1. Select neurons independently and exclusively based on the mutation ratio for each layer.  
   2. Switch (shuffle) selected neurons. If weights are stored in a matrix, NS switch rows without altering the order of elements within each row.  
       
   Input: trained model and mutation ratio  
   Output: mutated trained model   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.NS_mut(model, mutation_ratio)
   ```
   Example:  
   ```python
   NS_model = model_mut_opts.NS_mut(model, 0.01)
   ```
   
   Remarks that since NS mutation operator should generate various mutant based on different mutation ratio given. If you switch neurons multiple times, it's the same effect of shuffle a portion of neurons.
   
-  <b>LD - Layer Deactivation:</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Deactivate the effects of a randomly selected layer which satisfies conditions    
   Implementation:  
   1. LD operator traverses through the entire structure of deep learning model and record all the layers which are suitable.  Note that simply removing a layer from a trained deep learning model can break the model structure. LD is restricted to mutate layers whose input and output shapes are consistent.  
   2. One of the selected layers is randomly removed from the deep learning model.  
           
   Input: trained model  
   Output: mutated trained model   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.LD_mut(model)
   ```
   Example:  
   ```python
   LD_model = model_mut_opts.LD_mut(model)
   ```
   
   Remarks that the input and output layer should not be removed since the removal of input and output mutate the model to which totally different from the original one.  
   
-  <b>LAm - Layer Addition (model-level):</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Randomly Add (copy) a layer of previous layer to one of suitable spots in deep learning model      
   Implementation:  
   1. LAm operator traverses through the entire structure of deep learning model and record all the spots where a layer can be added. The condition is that the shape of input and output should be consistent to avoid breaking the original DNNs.  
   2. A layer is randomly added in one of the suitable spots.  
   
   Input: trained model  
   Output: mutated trained model   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.LAm_mut(model)
   ```
   Example:  
   ```python
   LAm_model = model_mut_opts.LAm_mut(model)
   ```
   
    Remarks LAm is quite similar to LAs operator, both of them add a layer within the deep learning constriction where the input and output must be the same. The only difference is that the weights of the newly added layer in LAm need to be copied from the previous layer.  
   
-  <b>AFRm - Activation Function Removal (model-level):</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Remove activation functions of a randomly selected layer    
   Implementation:  
   1. AFRm operator traverses through the entire structure of deep learning model and record all the layers with activation functions except the output layer.  
   2. AFRm randomly remove all activation functions of a randomly selected layer.  
      
   Input: trained model  
   Output: mutated trained model   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.AFRm_mut(model)
   ```
   Example:  
   ```python
   AFRm_model = model_mut_opts.AFRm_mut(model)
   ```
   
   Remarks that in my implementation, the activation functions of the output layer will not be included in the consideration. For instance, the value after activation function, softmax, of the output layer reflects the level of confidence. It may be better not to eliminate the activation functions of the output layer.
.  
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
