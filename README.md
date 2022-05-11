# **EvoMLP**
MLP without backpropagation

### **Quick theory behind the model**
This regressor and classifier model is fitted using N nets where the weights of the best N/4 nets repeatedly are paired into six new nets with a small random number added to the resulting nets weights. 

In the world of regressors and classifiers this model is problably not adding anything of value, but I thought it was a cool project and it turned out way more capable than I first thought.


### **API, regressor**
**Parameters**
**n : *int, default = 24***. 
Number of nets in the model. Will be rounded to the nearest 8th.


**hidden_layer_sizes : *List of ints, default = False***  
Numbers of nodes in the hidden layers. By default, there are no hidden layers in the nets.


**activation : *"relu", "leaky_relu", "sigmoid", default = "relu"***  
Activation function of the hidden layers.


**lr_target : *float, default = 0.002***  
Where standard deviance of the "mutation" will end.


**lr_initial_decay : *int, default = 20***  
How quick the inital "mutation" will decay.


**lr_final_decay : *float, default = 0.02***  
How quick the overall "mutation" will decay.


**random_state : *int, default = None***  
Value to seed NumPy with.



#### **.fit() parameters**
**X_train : *NumPy array of shape (samples, features)***


**y_train : *NumPy array of shape (samples, )***


**epochs : *int, default = 100***  
Number of iterations to fit the model.


**validation_data : *tuple of X_val and y_val, default = False***

**verbose : *{0, 1}, default = 0***  
0 : The fit process is quiet
1 : Each epoch with a new best net are printed out.


.fit() doesn't return anything.


#### **.predict() parameters**
**X : *NumPy array of shape (samples, features)***


.predict() returns a series of values predicted.
