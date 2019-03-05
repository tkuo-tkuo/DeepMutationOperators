# DeepMutation

This repository aims to implement mutation operators for Deep Learning Mutation Testing.  
  
The objective is to provide a tool for mutation testing on Deep Learning system. In this repository, 8 source-level mutation operators and 8 model level mutation operators will be implemented.  
  
The concept of these mutation operators is introduced in the paper, <b> DeepMutation: Mutation Testing of Deep Learning Systems </b>, where the link is attached in references. However, the coding implementation of each mutation operator is not explicitly explained. In this repository, we clarify the vague part and document how each mutation operator is actually implemented, aiming to present a convenient tool for mutation testing on Deep Learning system.  


Model-level mutation operators 
------------------
Model-level mutation operators directly mutate the structure and weights of DNN without training procedure, which is more efficient for mutated model generation. Model-level mutation operators automatically analysis structure of given DNN, mutate on a copy of the original DNN, and return the mutated copy of the original DNN.  
  
-  <b>GF - Gaussian Fuzzing:</b>  
   Target: Trained model (Weight)  
   Brief Operator Description: Fuzz a portion of weights in trained model   
   Implementation:  
   1. For weights of each layer, GF flattens weights of each layer to a one-dimensional list, since GF does not need to recognize the relationship between neurons. A one-dimensional list will be handy for manipulation.  
   2. GF mutation operator chooses elements for further mutation among the one-dimensional list independently and exclusively based on mutation ratio.  
   3. According to the type of distribution and corresponding user-configurable input parameters, GF mutation operator add noise on selected weights.
   
   Syntax:  
   ```python
    mutated_model  = model_mut_opts.GF_mut(model, mutation_ratio, prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None, mutated_layer_indices=None)
   ```
   Example:  
   ```python
    # Without specification of standard deviation parameter and type of probability distribution, normal distribution is used and STD is set to 0.1 as default
    GF_model = model_mut_opts.GF_mut(model, 0.01)
    # With specification of probability distribution type as normal distribution and STD value as 2
    GF_model = model_mut_opts.GF_mut(model, 0.01, prob_distribution='normal', STD=2)
    # With specification of probability distribution type as uniform distribution and corresponding lower and upper bound
    GF_model = model_mut_opts.GF_mut(model, 0.01, prob_distribution='uniform', lower_bound=-1, upper_bound=1)
    
    # Users can also indicate the indices of layers to be mutated
    GF_model = model_mut_opts.GF_mut(model, 0.01, prob_distribution='normal', STD=2, mutated_layer_indices=[0, 1])
   ```
   
   Remarks:  
   1. GF mutation operator extracts and mutates weights of Dense and Conv2D layer. For other types of layers like Activation, BatchNormalization, and Maxpooling, GF mutation operator will simply ignore these layers.  
   2. If needed, more type of probability distribution will be added. For instance, double-sided exponential distribution.  
   
-  <b>WS - Weight Shuffling:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Shuffle weights of selected neurons. which connect to previous layer   
   Implementation:  
   1.  Except for the input layer, for each layer, select neurons independently and exclusively based on the mutation ratio 
   2.  Shuffle the weights of each neuron's connections to the previous layer. For instance, the weights of Dense layer are stored in a matrix (2-dimension list) m * n. If neuron j is selected, all the weights w[:, j] connecting to neuron j are extracted, shuffled, and injected back in a matrix.  
   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.WS_mut(model, mutation_ratio, mutated_layer_indices=None)
   ```
   Example:  
   ```python
   WS_model = model_mut_opts.WS_mut(model, 0.01)
   
   # Users can also indicate the indices of layers to be mutated
   WS_model = model_mut_opts.WS_mut(model, 0.01, mutated_layer_indices=[0, 1])
   ```
   
   Remarks:
   1. Biases are excluded from consideration.  
   2. WS mutation operator extracts and mutates weights of Dense and Conv2D layer. For other types of layers like Activation, BatchNormalization, and Maxpooling, GF mutation operator will simply ignore these layers.  
   3. For Conv2D layer, since all neurons share the weights of filters during convolution, WS mutation operator shuffles the weights in selected output channels (filters) instead of selected neurons. 
     
-  <b>NEB - Neuron Effect Block:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Block effect of selected neurons on the next layer    
   Implementation:  
   1. Except for the output layer, for each layer, select neurons independently and exclusively based on the mutation ratio.  
   2. Block the effect of selected neurons by setting all the weights as 0.  
   
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.NEB_mut(model, mutation_ratio, mutated_layer_indices=None)
   ```
   Example:  
   ```python
   NEB_model = model_mut_opts.NEB_mut(model, 0.01)
   
   # Users can also indicate the indices of layers to be mutated
   NEB_model = model_mut_opts.NEB_mut(model, 0.01, mutated_layer_indices=[0, 1])
   ```
   
   Remarks:
   1. Biases are excluded for consideration.
   2. NEB mutation operator extracts and mutates weights of Dense and Conv2D layer. For other types of layers like Activation, BatchNormalization, and Maxpooling, NEB mutation operator will simply ignore these layers.  
   3. For Conv2D layer, since all neurons share the weights of filters during convolution, NEB mutation operator blocks effect in terms of selected input channels instead of selected neurons.
   
-  <b>NAI - Neuron Activation Inverse:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Invert (the sign) of activation status of selected neurons    
   Implementation:  
   1. Except for the input layer, for each layer, select neurons independently and exclusively based on the mutation ratio.  
   2. According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, invertion of the activation status of a neuron can be achieved by changing the sign of a neuron's value before applying its activation function. This can be actually achieved by multiplying -1 to all the weights connecting on the previous layer of selected neurons.
      
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.NAI_mut(model, mutation_ratio, mutated_layer_indices=None)
   ```
   Example:  
   ```python
   NAI_model = model_mut_opts.NAI_mut(model, 0.01)
   
   # Users can also indicate the indices of layers to be mutated
   NAI_model = model_mut_opts.NAI_mut(model, 0.01, mutated_layer_indices=[0, 1])
   ```
   
   Remarks:
   1. NAI mutation operator extracts and mutates weights of Dense and Conv2D layer. For other types of layers like Activation, BatchNormalization, and Maxpooling, NAI mutation operator will simply ignore these layers.
   2. For Conv2D layer, since all neurons share the weights of filters during convolution, NAI mutation operator mutates the weights of selected output channels (filters) instead of selected neurons.
   
-  <b>NS - Neuron Switch:</b>  
   Target: Trained model (Neuron)  
   Brief Operator Description: Switch two neurons (shuffle neurons) of the same layer  
   Implementation:  
   1. Select neurons independently and exclusively based on the mutation ratio for each layer.  
   2. Switch (shuffle) selected neurons, exchange their roles and influences on the next layer. 
          
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.NS_mut(model, mutation_ratio, mutated_layer_indices=None)
   ```
   Example:  
   ```python
   NS_model = model_mut_opts.NS_mut(model, 0.01)
   
   # Users can also indicate the indices of layers to be mutated
   NS_model = model_mut_opts.NS_mut(model, 0.01, mutated_layer_indices=[0, 1])
   ```
   
   Remarks:
   1. NS mutation operator extracts and mutates weights of Dense and Conv2D layer. For other types of layers like Activation, BatchNormalization, and Maxpooling, NS mutation operator will simply ignore these layers.
   2. For Conv2D layer, since all neurons share the weights of filters during convolution, NS mutation operator mutates the weights of selected input channels instead of selected neurons
   
-  <b>LD - Layer Deactivation:</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Deactivate the effects of selected layers which satisfies conditions    
   Implementation:  
   1. LD operator traverses through the entire structure of deep learning model and record all the suitable layers for LD mutation. Note that simply removing a layer from a trained deep learning model can break the model structure. LD is restricted to mutate layers whose input and output shapes are consistent.  
   2. If users do not indicate the indices of layers to be mutated, one of the suitable layers will be randomly selected and removed from the deep learning model.  
             
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.LD_mut(model, mutated_layer_indices=None)
   ```
   Example:  
   ```python
   LD_model = model_mut_opts.LD_mut(model)
   
   # Users can also indicate the indices of layers to be mutated
   LD_model = model_mut_opts.LD_mut(model, mutated_layer_indices=[0, 1])
   ```
   
   Remarks:
   1. The input and output layer should not be removed since the removal of input and output mutate the model to which totally different from the original one.  
   2. If the indices of layers indicated by users are invalid, LD will raise an error to notify the users. 
   
-  <b>LAm - Layer Addition (model-level):</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Add (copy) layers to suitable spots in deep learning model      
   Implementation:  
   1. LAm operator traverses through the entire structure of deep learning model and record all the suitable spots. There is a restricted condition that the shape of input and output should be consistent to avoid breaking the structure of original DNNs.  
   2. If users do not indicate the indices of layers to be added, a layer is randomly added in one of the suitable spots.     
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.LAm_mut(model, mutated_layer_indices=None)
   ```
   Example:  
   ```python
   LAm_model = model_mut_opts.LAm_mut(model)
   
   # Users can also indicate the indices of layers to be mutated
   LAm_model = model_mut_opts.LAm_mut(model, mutated_layer_indices=[0, 1])
   ```
   
    Remarks:
    1. The weights of the newly added layer in LAm are copied from the previous layer.  
   
-  <b>AFRm - Activation Function Removal (model-level):</b>  
   Target: Trained model (Layer)  
   Brief Operator Description: Remove activation functions of selected layers    
   Implementation:  
   1. AFRm operator traverses through the entire structure of deep learning model and record all the layers with activation functions except the output layer.  
   2. If users do not indicate the indices of layers to be added, AFRm randomly remove all activation functions of a layer from selected layers.  
       
   Syntax:  
   ```python
   mutated_model  = model_mut_opts.AFRm_mut(model, mutated_layer_indices=None)
   ```
   Example:  
   ```python
   AFRm_model = model_mut_opts.AFRm_mut(model)
   
   # Users can also indicate the indices of layers to be mutated
   AFRm_model = model_mut_opts.AFRm_mut(model, mutated_layer_indices=[0, 1])
   ```
   
   Remarks:
   1. In my implementation, the activation functions of the output layer will not be included in the consideration. For instance, the value after activation function, softmax, of the output layer reflects the level of confidence. It may be better not to eliminate the activation functions of the output layer.  

Source-level mutation operators 
------------------
Source-level mutation operators mutate either the original training dataset or the original training program. A training dataset mutant or training program mutant can further participate in the training process to generate a mutated model for mutation testing.  
  
For each of the mutation operators, there are several user-configurable parameters can be specified. See the description of individual operators Implementation for more details.  
  
-  <b> DR - Data Repetition:</b>  
   Target : Training dataset  
   Brief Operator Description: DR operator duplicates a portion of training dataset  
   Implementation:  
   1. A portion of samples among dataset will be chosen independently and exclusively based on mutation ratio.  
   2. Selected samples will be duplicated.  
   3. Duplicated samples will be concatenated with the original training dataset.  
   
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
   Brief Operator Description: LE operator falsifies a portion of results (e.g., labels) in traning dataset    
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusively based on mutation ratio.  
   2. Each result (e.g., label) among the chosen samples is mislabeled by LE operator. For instance, if the set of labels is donated as L, {0, 1, ..., 9}, and the correct label is 0, LE operator will randomly assign a value among L except the correct label 0.    
  
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
   1. A specific amount of samples is chosen independently and exclusively for further removal based on mutation ratio.  
   2. Selected samples in the training dataset are removed.  
      
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
   Brief Operator Description: Shuffle a portion of samples among training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusivel based on mutation ratio.  
   2. Selected samples will be extracted, shuffled, and injected back to the training dataset.  
   
   Syntax:  
   ```python
    mutated_dataset, copied_model  = source_mut_opts.DF_mut(training_dataset, model, mutation_ratio)
   ```
   Example:  
   ```python
    (DF_train_datas, DF_train_labels), DF_model = source_mut_opts.DF_mut((train_datas, train_labels), model, 0.01)
   ```
   
   Remarks:
   1. Only the selected samples will be shuffled and the order of unselected samples is preserved.  
   
-  <b>NP - Noise Perturb:</b>  
   Target: Training dataset  
   Brief Operator Description: Add noise to a portion of samples among training dataset  
   Implementation:  
   1. A specific amount of samples is chosen independently and exclusivel based on mutation ratio.  
   2. Noises are appended on each of the selected datasets. Since raw data in the training dataset and test dataset has been standardized with 0 mean and unit standard deviation, the value of noises follows normal distribution, where standard deviation parameter is a user-configurable parameter with default value 0.1 and mean is 0.    
       
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
   
   Syntax:  
   ```python
    copied_dataset, mutated_model  = source_mut_opts.LR_mut(training_dataset, model)
   ```
   Example:  
   ```python
    (LR_train_datas, LR_train_labels), LR_model = source_mut_opts.LR_mut((train_datas, train_labels), model)
   ```
  
   Remarks:
   1. In my implementation, the input layer and output layer will not be included in the consideration.   
   
-  <b>LAs - Layer Addition for source-level:</b>  
   Target: Training program  
   Brief Operator Description: Randomly add a layer to one of suitable spots in the deep learning model  
   Implementation:  
   1. LAs operator traverses through the entire structure of deep learning model and record all the spots where a layer can be added.   
   2. According to the paper, DeepMutation: Mutation Testing of Deep Learning Systems, LAs operator mainly focuses on adding layers like Activation, BatchNormalization. More types of layers should be considered in the future implementation once addition of a layer will not generate obviously different Deep Learning model from the original one, where unqualified mutant can be filtered out.   
  
   Syntax:  
   ```python
    copied_dataset, mutated_model  = source_mut_opts.LAs_mut(training_dataset, model)
   ```
   Example:  
   ```python
    (LAs_train_datas, LAs_train_labels), LAs_model = source_mut_opts.LAs_mut((train_datas, train_labels), model)
   ```
  
   Remarks:
   1. In my implementation, the output layer will not be included in the consideration.   

-  <b>AFRs - Activation Function Removal for source-level:</b>  
   Target: Training program  
   Brief Operator Description: Remove activation layers of a randomly selected layer    
   Implementation:  
   1. AFRs operator traverses through the entire structure of deep learning model and record all the layers with activation functions except the output layer.  
   1. AFRs randomly remove all activation functions of a randomly selected layer.  
     
   Syntax:  
   ```python
    copied_dataset, mutated_model  = source_mut_opts.AFRs_mut(training_dataset, model)
   ```
   Example:  
   ```python
    (AFRs_train_datas, AFRs_train_labels), AFRs_model = source_mut_opts.AFRs_mut((train_datas, train_labels), model)
   ```
   
   Remarks:
   1. In my implementation, the activation functions of the output layer will not be included in the consideration. For instance, the value after activation function, softmax, of the output layer reflects the level of confidence. It may be better not to eliminate the activation functions of the output layer.  
   

Assumption & Suggestion of Usage
----------------
<b>Assumption</b>
- Results (labels) of dataset are assumed to be one-hot encoded.  
- Currently, this DeepMutation API is mainly designed for fully-connected neural netowrks and convolutional neural networks. Models inputed are assumed to be either fully-connect neural networks or convolutional neural networks.   

<b>Suggestion</b>
- While constructing the architecture of deep neural networks, users should indicate the input shape for each layer if possible.   

 
Purpose & Content of each file  
----------------
Files below are ordered in alphabetical order.  
-  <b>example_model_level.ipynb</b>  
   This file illustrates the usage of source-level mutation operators, where usage of each mutation operator is separated into blocks for a better demonstration.   
   
-  <b>example_source_level.ipynb</b>  
   This file illustrates the usage of model-level mutation operators, where usage of each mutation operator is separated into blocks for a better demonstration.  
   
-  <b>model_mut_model_generators.py</b>  
   This file provides an interface for generating mutated models by model-level mutation operators. It is implemented for my own convenience to facilitate debugging and experiments. Users can solely use mutation operators without the usage of this file.  
   
-  <b>model_mut_operators.py</b>  
   This file consists of two classes, ModelMutationOperators and ModelMutationOperatorsUtils. ModelMutationOperators represents the logic of mutation for each model-level mutation operator, whereas ModelMutationOperatorsUtils extracts all tedious manipulation out from ModelMutationOperators to keep the code readable and maintainable. The class ModelMutationOperators is the interface for users to directly mutate their trained models.  
   
-  <b>network.py</b>  
   This file encapsulates functionalities related to neural network training into two classes, FCNetwork and CNNNetwork.  For instance, dataset processing, model compilation, training process, and model evaluation. Two fully-connected neural networks and two convolutional neural networks are provided respectively in class FCNetwork and class CNNNetwork. However, this file is implemented for facilitating debugging and experiments for my own convenience. Users can use their own neural network architectures, hyperparameters configuration, and datasets.  
   
-  <b>source_mut_model_generators.py</b>  
   This file provides an interface for generating mutated models by source-level mutation operators. It is implemented for my own convenience to facilitate debugging and experiments. Users can solely use mutation operators and train with either mutated dataset or mutated model by themselves without the usage of this file to generate mutated models.  
   
-  <b>source_mut_operators.py</b>  
   This file consists of two classes, SourceMutationOperators and SourceMutationOperatorsUtils. SourceMutationOperators represents the logic of mutation for each source-level mutation operator, whereas SourceMutationOperatorsUtils extracts all tedious manipulation out from SourceMutationOperators to keep the code readable and maintainable. The class SourceMutationOperators is the interface for users to mutate either their untrained models or datasets.  

-  <b>utils.py</b>  
   This file currently consists of three classes,  GeneralUtils, ModelUtils, and ExaminationalUtils.  
   
   1. GeneralUtils contains functions which are used repeatedly in other files. For instance, various shuffle functions.  
   2. ModelUtils contains functions related to neural network architectures and configurations which are used frequently for both source-level and model-level mutation operators. For instance, the model copy function, since TensorFlow does not actually support deep copy for model instance.  
   3. ExaminationalUtils contains functions which prevent invalid or problematic inputs. For instance, invalid mutation ratio which is out of the range between 0 and 1.  


Configuration
----------------
  Python: 3.5.1  
  Tensorflow: 1.11.0  
  Keras: 2.2.4  
  NumPy: 1.15.1  


References
----------------
  Lei Ma, Fuyuan Zhang, Jiyuan Sun, Minhui Xue, Bo Li, Felix Juefei-Xu, Chao Xie, Li Li, Yang Liu, Jianjun Zhao, et al. <br/>
  DeepMutation:  Mutation testing of Deep Learning Systems. <br/>
  https://arxiv.org/pdf/1805.05206.pdf <br/>

  Jingyi Wang, Guoliang Dongy, Jun Sun, Xinyu Wangy, Peixin Zhangy, Singapore University of Technology and Design Zhejiang University.<br/> 
  Adversarial Sample Detection for Deep Neural Network through Model Mutation Testing <br/>
  https://arxiv.org/pdf/1812.05793.pdf <br/>
