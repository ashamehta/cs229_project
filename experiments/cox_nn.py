
# from typing import Sequence

import tensorflow as tf
import keras as ks
import numpy as np
import sklearn as sk
from sksurv.metrics import concordance_index_censored
from sklearn import model_selection

import pandas as pd

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


def get_y_labels(y_event, y_time):
    """
    Formats the y_true labels to feed the Cox NN loss function.
    The first column of y_true would be the |y_events|, the second column would be |y_time|,
    and columns 3 through 3+num_samples would be the riskset matrix.
    """
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
    # option: normalize predictions

    event = tf.cast(event, predictions.dtype)
    riskset = tf.cast(riskset, predictions.dtype)

    predictions_masked = tf.math.multiply(tf.transpose(predictions), riskset)   # [num_samples, num_samples, sum across each column.
    rr = tf.math.reduce_logsumexp(predictions_masked, axis=1, keepdims=True)    # [num_samples, 1] each entry is rr of its sample
    assert rr.shape.as_list() == predictions.shape.as_list()

    losses = tf.math.multiply(event, rr - predictions)
    return losses   # option: add L1 or L2 regularization term?


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

def train_cox_nn_model(X_train, y_train, X_valid, y_valid, params):
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
            monitor="val_loss", min_delta=0, patience=15, mode="min", restore_best_weights=True)
    ]
    history = model.fit(
        X_train, y_labels_train, validation_data=(X_valid, y_labels_valid),
        batch_size=num_samples, epochs=params["num_epochs"], callbacks=callbacks)
    return model


def train_and_evaluate_cox_nn_model(params):
    """
    Gets the training, validation, and test data.
    Uses the data to train and evaluate the Cox PH neural network with the specified |params|.
    """
    # Get clinical and gene expression data.
    clinical_tsv = "processed_data/clinical_processed.tsv"
    gexp_top05_tsv = "processed_data/gene_expression_top05_matrix.tsv"
    cox_elast_gexp_models = "output/cox_model_elastic_gexp_exp3.tsv"

    clinical_df = pd.read_csv(clinical_tsv, sep="\t")
    gexp_df = pd.read_csv(gexp_top05_tsv, sep="\t")
    model_df = pd.read_csv(cox_elast_gexp_models, sep="\t", index_col=0)

    if params["num_features"] == "top5percent":
        selected_gexp_df = gexp_df
    else:
        selected_gexp_df = fs.select_features_from_cox_coef(model_df, gexp_df, num_features=params["num_features"])

    # Organize survival data into splits and get relevant data.
    cox_data = cox.CoxRegressionDataset(selected_gexp_df, clinical_df, standardize=True)
    X_train, X_valid, y_train, y_valid =\
        model_selection.train_test_split(cox_data.X, cox_data.y, test_size=params["validation_split"], shuffle=False)
    X_test, y_test = cox_data.X_test, cox_data.y_test

    event_field, time_field = y_train.dtype.names
    y_event_train, y_time_train = y_train[event_field], y_train[time_field]     # 98 number of censored samples
    y_event_valid, y_time_valid = y_valid[event_field], y_valid[time_field]
    y_event_test, y_time_test = y_test[event_field], y_test[time_field]

    y_labels_train = get_y_labels(y_event_train, y_time_train)
    y_labels_valid = get_y_labels(y_event_valid, y_time_valid)
    y_labels_test = get_y_labels(y_event_test, y_time_test)

    # Train model. Use validation data to determine convergence.
    model = train_cox_nn_model(X_train, y_train, X_valid, y_valid, params)

    # Evaluate model with validation data.
    print("\nValidation Data:")
    num_samples = X_valid.shape[0]
    valid_metrics = model.evaluate(X_valid, y_labels_valid, batch_size=num_samples)
    # print("Validation metrics:", metrics)

    # Evaluate model with test data.
    print("\nTest Data:")
    num_samples = X_test.shape[0]
    test_metrics = model.evaluate(X_test, y_labels_test, batch_size=num_samples)
    # test_predictions = model.predict(X_test, batch_size=num_samples)
    # c_index = concordance_index_censored(y_test[event_field], y_test[time_field], test_predictions[:,0])[0]
    # print("Test Concordance Index Censored:", c_index)

    return valid_metrics, test_metrics, history


def run_trials(params, num_trials=10):
    """
    Run the train-evaluate-test cycle multiple times to evaluate this model-training procedure.
    """
    best_valid, best_trial = 0, None
    valid_metrics, test_metrics = [], []
    for i in range(num_trials):
        valid, test, history = train_and_evaluate_cox_nn_model(params)
        valid_metrics.append(valid[1])
        test_metrics.append(test[1])
        if valid[1] > best_valid:
            best_valid, best_trial = valid[1], i
    print("\nValidation:\tBest=%s\tAverage=%s\t%s" % (best_valid, np.mean(valid_metrics), valid_metrics))
    print("\nTest:\t\tBest=%s\tAverage=%s\t%s" % (test_metrics[best_trial], np.mean(test_metrics), test_metrics))
    return valid_metrics, test_metrics




################################################################
# Experiment scripts.
################################################################

# Run 1: Uses only 79 top-ranked features, some basic regularization with 50 latent neurons.
params1 = {
    "validation_split": 0.25,
    "num_features": 79,
    "num_latent_neurons": 50,
    "l1_kernel_regularizer": 0,
    "l2_kernel_regularizer": 1e-4,
    "dropout_rate": 0.1,
    "num_epochs": 150,  # EarlyStop callback is expected to terminate run early.
}
print("\nRUN 1:\n", params1)
run_trials(params1)

# RUN 1
# Validation:     Best=0.7416038513183594 Average=0.713913643360138       [0.6970527768135071, 0.7004798054695129, 0.7416038513183594, 0.7100753784179688, 0.7128170132637024, 0.703906774520874, 0.7292666435241699, 0.7244688272476196, 0.6991089582443237, 0.7203564047813416]
# Test:           Best=0.5826566219329834 Average=0.6014211118221283      [0.5768561363220215, 0.5968677401542664, 0.5826566219329834, 0.6125289797782898, 0.6041183471679688, 0.6255800724029541, 0.5971577763557434, 0.636310875415802, 0.6186195015907288, 0.5635150671005249]


# Run 2: Uses all of the top 5% of features, with some basic regularization and 50 latent neurons.
params2 = {
    "validation_split": 0.25,
    "num_features": "top5percent",
    "num_latent_neurons": 100,
    "l1_kernel_regularizer": 1e-4,
    "l2_kernel_regularizer": 1e-3,
    "dropout_rate": 0.3,
    "num_epochs": 200,  # EarlyStop callback is expected to terminate run early.
}
# print("\nRUN 2:\n", params2)
# run_trials(params2)













# Ensemble Method: categorize sample, and train a model for each category

# Which features most important? Why might they be most important?

# Correlation between feature and predicted T values?

# Covariate: age, stage of cancer at diagnosis

