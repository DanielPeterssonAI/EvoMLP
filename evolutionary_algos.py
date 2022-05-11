import numpy as np
import math

class EvoMLPRegressor:
    def __init__(
        self, 
        n = 24, 
        hidden_layer_sizes = False, 
        activation = "relu", 
        lr_target = 0.002, 
        lr_initial_decay = 20, 
        lr_final_decay = 0.02, 
        random_state = None
    ):
        self.n = int(round(n / 8) * 8)
        self.validation_loss_history = []
        self.training_loss_history = []
        self.random_state = random_state
        self.activation = activation
        self.lr_target = lr_target
        self.lr_initial_decay = lr_initial_decay
        self.lr_final_decay = lr_final_decay
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X_train, y_train, epochs = 100, validation_data = False, verbose = 0):

        if self.random_state != None:
            np.random.seed(self.random_state)

        if validation_data:
            X_val, y_val = validation_data

        if self.activation == "sigmoid":
            activation_function = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation == "leaky_relu":
            activation_function = lambda x: np.maximum(0.1 * x, x)
        elif self.activation == "relu":
            activation_function = lambda x: np.maximum(0, x)
        
        # Add bias column to X
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        
        if self.hidden_layer_sizes:
            layers = [X_train.shape[1]] + self.hidden_layer_sizes + [1]
        else:
            layers = [X_train.shape[1]] + [1]
        
        n = self.n
        ndiv4 = n // 4
        number_of_layers_minus_one = len(layers) - 1

        lr_target = self.lr_target
        lr_initial_decay = self.lr_initial_decay
        lr_final_decay = self.lr_final_decay
        
        y_preds = np.zeros((n, y_train.shape[0]))
        nets_loss = np.zeros(n)
        sorted_indices = np.arange(-(ndiv4), n, 1)

        best_net_index = -1
        weights = []

        for i in range(number_of_layers_minus_one):
            weights += [np.random.normal(0, 2, (n, layers[i], layers[i + 1]))]

        for epoch in range(epochs):
            forward_pass = X_train.T
            
            # Hidden layers
            for j in range(number_of_layers_minus_one - 1):
                forward_pass = activation_function(weights[j][sorted_indices[ndiv4:]].transpose(0, 2, 1) @ forward_pass)

            # Output layer
            forward_pass = weights[-1][sorted_indices[ndiv4:]].transpose(0, 2, 1) @ forward_pass
            
            # Fill in in the predictions from the new nets
            y_preds[sorted_indices[ndiv4:]] = forward_pass.reshape(*forward_pass.shape[::2])
            nets_loss[sorted_indices[ndiv4:]] = np.mean(np.abs(y_preds[sorted_indices[ndiv4:]] - y_train), axis = 1)
            
            sorted_indices = np.argsort(nets_loss)

            mutation_sigma = math.exp(-epoch / (epochs / (lr_initial_decay * math.log10(epochs + 1)))) + lr_final_decay * math.exp(-(epoch + 1) * (1 / (epochs))) + lr_target + (-0.035 * 10 * lr_final_decay)

            for j in range(number_of_layers_minus_one):
                weights[j][sorted_indices[0 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[1 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[2 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[3 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[4 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[5 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))

            if best_net_index != sorted_indices[0]:
                best_net_index = sorted_indices[0]
                self.training_loss_history += [nets_loss[best_net_index]]

                self.best_net_weights = []
                for j in range(number_of_layers_minus_one):
                    self.best_net_weights += [weights[j][best_net_index]]
                
                if validation_data:
                    self.validation_loss_history += [np.mean(np.abs(y_val - self.predict(X_val)))]
                    if verbose == 1:
                        print(f"Epoch {epoch} - loss: {self.training_loss_history[-1]} - val_loss: {self.validation_loss_history[-1]} - sigma: {mutation_sigma}")
                else:
                    if verbose == 1:
                        pass
                        print(f"Epoch {epoch} - loss: {self.training_loss_history[-1]} - sigma: {mutation_sigma}")


    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]

        if self.activation == "sigmoid":
            activation_function = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation == "leaky_relu":
            activation_function = lambda x: np.maximum(0.1 * x, x)
        else:
            activation_function = lambda x: np.maximum(0, x)

        forward_pass = X.T
        for j in range(len(self.best_net_weights) - 1):
            forward_pass = activation_function(self.best_net_weights[j].T @ forward_pass)

        forward_pass = self.best_net_weights[-1].T @ forward_pass
        return forward_pass.ravel()


class EvoMLPClassifier:
    '''LATEST VERSION, TESTED WITH MULTICLASS AND BINARY'''

    def __init__(self, n = 24, hidden_layers = False, activation = "relu", lr_target = 0.04, lr_initial_decay = 60, lr_final_decay = 0.03, random_state = None):

        self.n = int(round(n / 8) * 8)
        self.validation_loss_history = []
        self.training_loss_history = []
        self.random_state = random_state
        self.activation = activation
        self.lr_target = lr_target
        self.lr_initial_decay = lr_initial_decay
        self.lr_final_decay = lr_final_decay

        
        if hidden_layers:
            self.hidden_layers = hidden_layers
        else:
            self.hidden_layers = False

        

    def fit(self, X_train, y_train, epochs = 100, validation_data = False, verbose = 0):

        n = self.n
        ndiv4 = n // 4

        if self.random_state != None:
            np.random.seed(self.random_state)

        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        y_train = y_train.astype("int8")

        if len(y_train.shape) == 1:
            self.multiclass = False
        elif len(y_train.shape) == 2 and y_train.shape[1] == 1:
            self.multiclass = False
            y_train = y_train.ravel()
        else:
            self.multiclass = True
            

        if validation_data:
            X_val, y_val = validation_data

        if self.activation == "sigmoid":
            activation_function = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation == "leaky_relu":
            activation_function = lambda x: np.maximum(0.1 * x, x)
        else:
            activation_function = lambda x: np.maximum(0, x)

        if self.multiclass == True:
            output_activation_function = lambda x: np.exp(x) / np.sum(np.exp(x), axis = 2, keepdims = True)
            
            def loss_function(y_train, y_preds, sorted_indices):
                return np.mean(np.sum(-y_train * np.log10(y_preds[sorted_indices[ndiv4:]]), axis = 2), axis = 1)

        elif self.multiclass == False:
            output_activation_function = lambda x: (1 / (1 + np.exp(-x))).reshape(x.shape[:2])

            def loss_function(y_train, y_preds, sorted_indices):
                return np.mean(np.abs(y_preds[sorted_indices[ndiv4:]] - y_train), axis = 1)

        lr_target = self.lr_target
        lr_initial_decay = self.lr_initial_decay
        lr_final_decay = self.lr_final_decay

        layers = [X_train.shape[1]]

        if self.hidden_layers:
            layers = [X_train.shape[1]] + self.hidden_layers

        if self.multiclass == True:
            layers = layers + [y_train.shape[1]]
        elif self.multiclass == False:
            layers = layers + [1]

        number_of_layers_minus_one = len(layers) - 1
        
        if self.multiclass == True:
            y_preds = np.zeros((n, y_train.shape[0], y_train.shape[1]))
        elif self.multiclass == False:
            y_preds = np.zeros((n, y_train.shape[0]))

        nets_loss = np.zeros(n)
        sorted_indices = np.arange(-(ndiv4), n, 1)

        best_net_index = -1

        weights = []

        for i in range(number_of_layers_minus_one):
            weights += [np.random.normal(0, 1, (n, layers[i], layers[i + 1]))]

        for epoch in range(epochs):
            forward_pass = X_train.T

            
            for j in range(number_of_layers_minus_one - 1):
                forward_pass = activation_function(weights[j][sorted_indices[ndiv4:]].transpose(0, 2, 1) @ forward_pass)
            
            forward_pass = weights[-1][sorted_indices[ndiv4:]].transpose(0, 2, 1) @ forward_pass
            
            y_preds[sorted_indices[ndiv4:]] = output_activation_function(forward_pass.transpose(0, 2, 1))

            nets_loss[sorted_indices[ndiv4:]] = loss_function(y_train, y_preds, sorted_indices)

            sorted_indices = np.argsort(nets_loss)
            mutation_sigma = math.exp(-epoch / (epochs / (lr_initial_decay * math.log10(epochs + 1)))) + lr_final_decay * math.exp(-(epoch + 1) * (1 / (epochs))) + lr_target + (-0.036 * 10 * lr_final_decay)

            for j in range(number_of_layers_minus_one):
                weights[j][sorted_indices[0 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[1 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[2 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[3 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[4 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))
                weights[j][sorted_indices[5 + ndiv4::6]] = (weights[j][sorted_indices[0: ndiv4: 2]] + weights[j][sorted_indices[1: ndiv4: 2]]) / 2 + np.random.normal(0, mutation_sigma, (ndiv4 // 2, layers[j], layers[j + 1]))

            if best_net_index != sorted_indices[0]:
                best_net_index = sorted_indices[0]
                self.training_loss_history += [nets_loss[best_net_index]]
                

                self.best_net_weights = []
                for j in range(number_of_layers_minus_one):
                    self.best_net_weights += [weights[j][best_net_index]]
                
                if validation_data:
                    self.validation_loss_history += [np.mean(np.abs(y_val - self.predict(X_val)))]
                    if verbose == 1:
                        print(f"Epoch {epoch} - loss: {self.training_loss_history[-1]} - val_loss: {self.validation_loss_history[-1]}")
                else:
                    if verbose == 1:
                        pass
                        print(f"Epoch {epoch} - loss: {self.training_loss_history[-1]} - {mutation_sigma}")


    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]

        if self.activation == "sigmoid":
            activation_function = lambda x: 1 / (1 + np.exp(-x))
        elif self.activation == "leaky_relu":
            activation_function = lambda x: np.maximum(0.1 * x, x)
        else:
            activation_function = lambda x: np.maximum(0, x)

        if self.multiclass == True:
            output_activation_function = lambda x: np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

        elif self.multiclass == False:
            output_activation_function = lambda x: (1 / (1 + np.exp(-x))).reshape(x.shape[:1])

        forward_pass = X.T

        for j in range(len(self.best_net_weights) - 1):
            forward_pass = activation_function(self.best_net_weights[j].T @ forward_pass)
            
        forward_pass = self.best_net_weights[-1].T @ forward_pass
            
        return output_activation_function(forward_pass.T)
