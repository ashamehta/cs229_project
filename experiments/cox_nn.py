
# from typing import Sequence

import tensorflow as tf
import keras as ks
import sklearn as sk
from sksurv.metrics import concordance_index_censored
from sklearn import model_selection


import numpy as np
import pandas as pd
import itertools

import cox_regression_main as cox
import feature_selection as fs

################################################################
# Utility functions.
################################################################

def compute_riskset_matrix(time):
    """
    Compute mask that represents each sample's risk set. Uses Breslow's method for tie-breakers.
    The risk set would be a boolean (n_samples, n_samples) matrix where the `i`-th row denotes the
    risk set of the `i`-th instance, i.e. the indices `j` for which the observer time `y_j >= y_i`.
    """
    # Sort in descending order.
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for ordered_i, time_i in enumerate(o):
        ti = time[time_i]
        k = ordered_i
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[time_i, o[:k]] = True
    return risk_set


def get_y_labels(y_raw):
    """
    Formats the y_true labels to feed the Cox NN loss function.
    The first column of y_true would be the events of |y_raw|, the second column would be the
    times of |y_raw|, and columns 3 through 3+num_samples would be the riskset matrix.
    """
    event_field, time_field = y_raw.dtype.names
    y_event, y_time = y_raw[event_field], y_raw[time_field]

    riskset_matrix = compute_riskset_matrix(y_time)
    events = np.array([y_event]).T
    time = np.array([y_time]).T
    return np.concatenate((events, time, riskset_matrix), axis=1).astype(int)


def cox_ph_loss(y_true, y_pred):
    """
    Computes the partial log likelihood of the Cox PH Regression model.
    Assumes y_true is formatted as done by get_y_labels() above.
    """
    event, riskset = y_true[:,0:1], y_true[:,2:]    # tensors of shape [num_samples, 1] and [num_samples, num_samples]
    predictions = y_pred                            # tensor of shape [num_samples, 1]
    # option: "safe" normalize predictions

    event = tf.cast(event, predictions.dtype)
    riskset = tf.cast(riskset, predictions.dtype)

    predictions_masked = tf.math.multiply(tf.transpose(predictions), riskset)   # [num_samples, num_samples, sum across each column.
    rr = tf.math.reduce_logsumexp(predictions_masked, axis=1, keepdims=True)    # [num_samples, 1] each entry is rr of its sample
    assert rr.shape.as_list() == predictions.shape.as_list()

    losses = tf.math.multiply(event, rr - predictions)
    return losses


def concordance_metric(y_true, y_pred):
    """
    Computes the concordance metric censored.
    Assumes y_true is formatted as done by get_y_labels() above.
    """
    event, time = tf.cast(y_true[:,0], bool), y_true[:,1]       # tensors of shape [num_samples, 1] and [num_samples, 1]
    predictions = y_pred[:,0]                                   # tensor of shape [num_samples, 1]
    return concordance_index_censored(event.numpy(), time.numpy(), predictions.numpy())[0]




################################################################
# Experiment routines.
################################################################

def get_raw_data(num_features=79, test=False):
    """
    Get the data to be used for the Cox Neural Network.
    """
    # The data files to be used.
    suffix = "test" if test else "train"
    clinical_tsv = "processed_data/clinical_%s.tsv" % suffix    
    gexp_top05_tsv = "processed_data/gene_expression_top05_%s.tsv" % suffix
    # cox_elast_gexp_models = "output/cox_model_elastic_gexp_exp5.tsv"
    cox_elast_gexp_models = "output/cox_model_elastic_si_gexp_exp1.tsv"

    # Read in the clinical and gene expression data.
    clinical_df = pd.read_csv(clinical_tsv, sep="\t")
    gexp_df = pd.read_csv(gexp_top05_tsv, sep="\t")
    model_df = pd.read_csv(cox_elast_gexp_models, sep="\t", index_col=0)

    # Select the top-N features as determined by the Cox Net process
    if num_features != "top5percent":
        gexp_df = fs.select_features_from_cox_coef(model_df, gexp_df, num_features=num_features)

    # Format the survival data. Note that data already is in train/test splits, so need to split again.
    cox_data = cox.CoxRegressionDataset(gexp_df, clinical_df, standardize=True, test_size=0.0)  
    return cox_data.X, cox_data.y


def train_cox_nn_model(X_train, y_labels_train, X_valid, y_labels_valid, params):
    """
    Creates single-latent-layer neural network with a Cox PH loss, parameterized with |params|.
    Trains the model on X_train,y_train, and validates the model with X_valid,y_valid.
    """
    # Construct neural network model.
    tf.config.experimental_run_functions_eagerly(True)
    model = ks.models.Sequential()
    model.add(ks.layers.GaussianNoise(0.01))
    model.add(ks.layers.Dense(
        params["num_latent_neurons"], activation='tanh', name="LatentLayer",     # cited 'tanh' as optimal activation function to use
        kernel_regularizer=ks.regularizers.l1_l2(l1=params["l1_kernel_regularizer"], l2=params["l2_kernel_regularizer"])))
    model.add(ks.layers.Dropout(params["dropout_rate"]))
    model.add(ks.layers.Dense(1))
    model.compile(optimizer="Adam", loss=cox_ph_loss, metrics=[concordance_metric], run_eagerly=True)
    # default hyperparams: learning_rate=0.001, epsilon=1e-07

    # Train model on training dataset.
    num_samples, num_features = X_train.shape[0], X_train.shape[1]
    print("Number of training samples:", num_samples)
    print("Number of features:", num_features)
    callbacks = [
        ks.callbacks.EarlyStopping(
            monitor="concordance_metric", min_delta=0, patience=20, mode="max", restore_best_weights=True)
    ]
    history = model.fit(
        X_train, y_labels_train, validation_data=(X_valid, y_labels_valid),
        batch_size=num_samples, epochs=params["num_epochs"], callbacks=callbacks)
    return model, history


def train_and_evaluate_cox_nn_model(X, y_raw, params):
    """
    Uses cross-validation to train and evaluate the Cox PH neural network with the specified |params|.
    """
    # X, y = data_splits["X_train"], data_splits["y_train"]

    kf = model_selection.KFold(n_splits=3)
    valid_scores, train_scores = [], []

    for train_index, valid_index in kf.split(X):
        # Compute the splits for this cross-validation split.
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = get_y_labels(y_raw[train_index]), get_y_labels(y_raw[valid_index])

        for trial in range(2):
            # Train model. Use validation data to determine convergence.
            model, history = train_cox_nn_model(X_train, y_train, X_valid, y_valid, params)
            valid_metrics = model.evaluate(X_valid, y_valid, batch_size=X_valid.shape[0])
            train_metrics = model.evaluate(X_train, y_train, batch_size=X_train.shape[0])

            valid_scores.append(valid_metrics[1])
            train_scores.append(train_metrics[1])

    valid_mean, train_mean = np.mean(valid_scores), np.mean(train_scores)
    return valid_mean, train_mean


def grid_search(X, y_raw, grid_params, default_params):
    """
    Does grid search over the possible values of the params in |grid_params| (a mapping from param to list of possible values).
    """
    variables = list(grid_params.keys())
    variable_values = [grid_params[p] for p in variables]
    print("Grid Search variables:", variables)

    grid_results = []
    best_valid, best_train, best_params = 0, 0, None
    for combo in itertools.product(*variable_values):
        print(combo)
        # assert False
        parameters = default_params.copy()
        for param, param_val in zip(variables, combo):
            parameters[param] = param_val
        
        print("Training NN on combo=", combo)
        valid_mean, train_mean = train_and_evaluate_cox_nn_model(X, y_raw, parameters)
        grid_results.append((combo, valid_mean, train_mean))

        if valid_mean > best_valid:
            best_valid, best_train, best_params = valid_mean, train_mean, parameters.copy()

    return (best_valid, best_train, best_params), grid_results


def run_grid_search_experiment(grid_params, params, num_test_trials=10):
    # Get training data.
    X_train, y_raw_train = get_raw_data(params["num_features"], test=False)

    # Use training data in grid search to get best parameters for training a model.
    best_try, grid_results = grid_search(X_train, y_raw_train, grid_params, params)
    best_valid, best_train, best_params = best_try[0], best_try[1], best_try[2]
    print("Grid Search complete.\nAve Cross-Valid Score=%s\nAve Train Score=%s\nSelected Params=%s\n" % (
        best_valid, best_train, best_params))

    run_experiment(best_params, num_test_trials)
    print("\nBest Parameters:")
    for grid_p in grid_params:
        print("  %s=%s" % (grid_p, best_params[grid_p]))
    print("\nAve Cross-Valid= %s" % best_valid)
    print(grid_results)
    return


def run_experiment(params, num_test_trials=10):
    # Get training data.
    X_train, y_raw_train = get_raw_data(params["num_features"], test=False)
    y_train = get_y_labels(y_raw_train)
    
    # Get test data.
    X_test, y_raw_test = get_raw_data(params["num_features"], test=True)
    y_test = get_y_labels(y_raw_test)

    # Get validation data for this run.
    # X_train, X_valid, y_raw_train, y_raw_valid = model_selection.train_test_split(X_train, y_raw_train, test_size=0.33, shuffle=False)
    # y_train, y_valid = get_y_labels(y_raw_train), get_y_labels(y_raw_valid)

    valid_scores, train_scores, test_scores = [], [], []
    for trial in range(num_test_trials):
        # print(X_train.shape, X_valid.shape, X_test.shape)
        # assert False
        # Train model on entire training dataset.
        model, history = train_cox_nn_model(X_train, y_train, X_train, y_train, params)
        
        train_metrics = model.evaluate(X_train, y_train, batch_size=X_train.shape[0])
        # valid_metrics = model.evaluate(X_valid, y_valid, batch_size=X_valid.shape[0])
        test_metrics = model.evaluate(X_test, y_test, batch_size=X_test.shape[0])
        
        train_scores.append(train_metrics[1])
        # valid_scores.append(valid_metrics[1])
        test_scores.append(test_metrics[1])

    best_i = np.argmax(test_scores)

    print("\nTrain:\tBest=%s\tAverage=%s\t" % (train_scores[best_i], np.mean(train_scores)))
    # print("\nValid:\tBest=%s\tAverage=%s\t" % (valid_scores[best_i], np.mean(valid_scores)))
    print("\nTest:\tBest=%s\tAverage=%s\t" % (test_scores[best_i], np.mean(test_scores)))
    return




################################################################
# Experiment scripts.
################################################################

# Experiment 1: Uses only 79 top-ranked features, some basic regularization with 50 latent neurons.
params1 = {
    "validation_split": 0.25,
    "num_features": 79,
    "num_latent_neurons": 50,
    "l1_kernel_regularizer": 0.001,
    "l2_kernel_regularizer": 0.0,
    "dropout_rate": 0.0,
    "num_epochs": 150,  # EarlyStop callback is expected to terminate run early.
}
# print("\nRUN 1:\n", params1)
# run_trials(params1)

# Train:  Best=0.7020984888076782 Average=0.7314346969127655
# Valid:  Best=0.6943090409040451 Average=0.7201890110969543
# Test:   Best=0.5512948036193848 Average=0.5443725168704987

# RUN 1
# Validation:     Best=0.7416038513183594 Average=0.713913643360138       [0.6970527768135071, 0.7004798054695129, 0.7416038513183594, 0.7100753784179688, 0.7128170132637024, 0.703906774520874, 0.7292666435241699, 0.7244688272476196, 0.6991089582443237, 0.7203564047813416]
# Test:           Best=0.5826566219329834 Average=0.6014211118221283      [0.5768561363220215, 0.5968677401542664, 0.5826566219329834, 0.6125289797782898, 0.6041183471679688, 0.6255800724029541, 0.5971577763557434, 0.636310875415802, 0.6186195015907288, 0.5635150671005249]



# Experiment 2: Using grid search, attempt to find the best number of L1-regularized latent neurons for the top ~80 features.
grid_params = {
    "num_latent_neurons": [55, 60, 65, 70], # 50
    "l1_kernel_regularizer": [0.1, 0.01, 0.001], # 0.0001
}
default_params = {
    "validation_split": 0.25,
    "num_features": 65, #120?
    "dropout_rate": 0.0,
    "l2_kernel_regularizer": 0.0,
    "num_epochs": 150
}
# run_grid_search_experiment(grid_params, default_params, num_test_trials=10)

# num_features=50
# Best Parameters (valid=0.7379134446382523, train=0.7115701138973236):
#   num_latent_neurons=70
#   l1_kernel_regularizer=0.001
# Train:  Best=0.7325053811073303 Average=0.7019871652126313
# Valid:  Best=0.7379134446382523 Average=0.698024046421051
# Test:   Best=0.5607569813728333 Average=0.5496514022350312

# num_features=55
# Best Parameters (valid=0.7572149634361267, train=0.7371134459972382):
#   num_latent_neurons=70
#   l1_kernel_regularizer=0.001
# Train:  Best=0.7402141094207764 Average=0.7262269794940949
# Valid:  Best=0.7572149634361267 Average=0.7158934652805329
# Test:   Best=0.5804283022880554 Average=0.5580179393291473

# num_features=60
# Best Parameters (valid=0.7377019375562668, train=0.7263097316026688):
#   num_latent_neurons=65
#   l1_kernel_regularizer=0.01
# Train:  Best=0.6967023611068726 Average=0.7095074951648712
# Valid:  Best=0.7377019375562668 Average=0.7097079038619996
# Test:   Best=0.5283864736557007 Average=0.5518924415111541

# num_features=70
# Best Parameters (valid=0.7160461843013763, train=0.7399457544088364):
#   num_latent_neurons=55
#   l1_kernel_regularizer=0.01
# Train:  Best=0.739700198173523  Average=0.7026638090610504
# Valid:  Best=0.7160461843013763 Average=0.6668384850025177
# Test:   Best=0.5906374454498291 Average=0.5474850684404373

# num_features=80
# Best Parameters (valid=0.7074719667434692, train=0.7389127463102341):
#   num_latent_neurons=55
#   l1_kernel_regularizer=0.01
# Train:  Best=0.7277087569236755 Average=0.7189807295799255
# Valid:  Best=0.7074719667434692 Average=0.6775773167610168
# Test:   Best=0.548804759979248  Average=0.5331175267696381




# Experiment 3: Use grid search with SI-CoxNet ranked features.
grid_params = {
    "num_latent_neurons": [55, 60, 65, 70], # 50
    "l1_kernel_regularizer": [0.001], # 0.01, 0.0001
    "dropout_rate": [0.0, 0.1, 0.2, 0.3],
}
default_params = {
    "validation_split": 0.25,
    "num_features": 65,
    # "dropout_rate": 0.1, # 0.0
    "l2_kernel_regularizer": 0.0,
    "num_epochs": 160
}
run_grid_search_experiment(grid_params, default_params, num_test_trials=10)



# num_features = 50
# Ave Cross-Valid= 0.6817124386628469
# Train:  Best=0.7296658754348755 Average=0.730408889055252
# Test:   Best=0.5871514081954956 Average=0.5726344645023346
# Best Parameters:
#   num_latent_neurons=65
#   l1_kernel_regularizer=0.01

# num_features = 55
# Ave Cross-Valid= 0.6984230677286783
# Train:  Best=0.7370467185974121 Average=0.7419573903083801
# Test:   Best=0.592131495475769  Average=0.5830677390098572
# Best Parameters:
#   num_latent_neurons=60
#   l1_kernel_regularizer=0.01
#   dropout_rate=0.0

# num_features = 55
# Ave Cross-Valid= 0.7059690952301025
# Train:  Best=0.7327658534049988 Average=0.7356246590614319
# Test:   Best=0.5933765172958374 Average=0.5687002003192901
# Best Parameters:
#   num_latent_neurons=55
#   l1_kernel_regularizer=0.001
#   dropout_rate=0.1

# num_features = 55
# Ave Cross-Valid= 0.6908576389153799
# Train:  Best=0.7406386733055115 Average=0.7208729028701782
# Test:   Best=0.5948705077171326 Average=0.5669322729110717
# Best Parameters:
#   num_latent_neurons=55
#   l1_kernel_regularizer=0.001
#   dropout_rate=0.3


# num_features = 60
# Ave Cross-Valid= 0.7021611432234446
# Train:  Best=0.7455592155456543 Average=0.7473306059837341
# Test:   Best=0.586902379989624  Average=0.5684760868549347
# Best Parameters:
#   num_latent_neurons=70
#   l1_kernel_regularizer=0.01

# num_features = 65
# Ave Cross-Valid= 0.7243321537971497
# Train:  Best=0.7649953365325928 Average=0.7646902561187744
# Test:   Best=0.5625     Average=0.5510707199573517
# Best Parameters:
#   num_latent_neurons=60
#   l1_kernel_regularizer=0.01
#   dropout_rate=0.0

# num_features = 65
# Ave Cross-Valid= 0.7133029500643412
# Train:  Best=0.7674555778503418 Average=0.7517000436782837
# Test:   Best=0.5891434550285339 Average=0.5483067750930786
# Best Parameters:
#   num_latent_neurons=60
#   l1_kernel_regularizer=0.001
#   dropout_rate=0.1 (out of 0.1, 0.2, 0.3)



# num_features=70
# Ave Cross-Valid= 0.7055005033810934
# Train:  Best=0.7748855948448181 Average=0.7688382565975189
# Test:   Best=0.5774402618408203 Average=0.5559760987758636
# Best Parameters:
#   num_latent_neurons=70
#   l1_kernel_regularizer=0.01






# Experiment 4: Using grid search, attempt to find the best dropout rate.
grid_params = {
    "num_latent_neurons": [70, 75, 80], # 50
    "l1_kernel_regularizer": [0.001, 0.001], # 0.0001
    "dropout_rate": [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
}
default_params = {
    "validation_split": 0.25,
    "num_features": 70, #120?
    "l2_kernel_regularizer": 0.0,
    "num_epochs": 150
}
# run_grid_search_experiment(grid_params, default_params, num_test_trials=10)

# Best Parameters (valid=0.7266203165054321, train=0.7270275205373764):
#   num_latent_neurons=70
#   l1_kernel_regularizer=0.001
#   dropout_rate=0.2
# Train:  Best=0.7211991548538208 Average=0.7242655277252197
# Valid:  Best=0.7266203165054321 Average=0.7011168301105499
# Test:   Best=0.573954164981842  Average=0.5640936255455017



# Experiment 5: Same thing, but with all top5% of features.
grid_params = {
    "num_latent_neurons": [40, 50, 60], # 50
    # "l1_kernel_regularizer": [0.001, 0.01], # 0.0001
    "dropout_rate": [0.3, 0.4, 0.5, 0.6, 0.7]
}
default_params = {
    "validation_split": 0.25,
    "num_features": "top5percent",
    "l1_kernel_regularizer": 0.001,
    "l2_kernel_regularizer": 0.0,
    "num_epochs": 150
}
# run_grid_search_experiment(grid_params, default_params, num_test_trials=10)




