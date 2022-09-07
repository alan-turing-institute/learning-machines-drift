"""TODO PEP 257"""
import numpy as np
import pandas as pd

from learning_machines_drift import FileBackend, Monitor, Registry, datasets
from learning_machines_drift.display import Display
from typing import Dict, Tuple

def generate_features_labels_latents(numrows:int)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    X, Y, latents =datasets.logistic_model(
        x_mu=np.array([0.0, 0.0, 0.0]), size=numrows, return_latents=True)
    
    features_df:pd.DataFrame = pd.DataFrame(
    {
        "age": X[:, 0],
        "height": X[:, 1],
        "ground-truth-label": Y
    })
    predictions_series:pd.Series = pd.Series(Y)
    latents_df:pd.DataFrame = pd.DataFrame({"latents": latents})
    return (features_df, predictions_series, latents_df)
    
def get_detector_reference(features_df:pd.DataFrame, predictions_series:pd.Series, latents_df:pd.DataFrame)->Registry:
    detector = Registry(tag="simple_example", backend=FileBackend("my-data"))
    detector.register_ref_dataset(
    features=features_df, labels=predictions_series, latents=latents_df)
    return detector

def log_new_data(detector:Registry, features_df:pd.DataFrame, predictions_series:pd.Series, latents_df:pd.DataFrame)->None:
    # Log features
    with detector:
        detector.log_features(features_df)
        detector.log_labels(predictions_series)
        detector.log_latents(latents_df)

def load_data()->Monitor:
    measure = Monitor(tag="simple_example", backend=FileBackend("my-data"))
    measure.load_data()
    return measure

def main():
    features_df, predictions_series, latents_df = generate_features_labels_latents(10)
    detector:Registry = get_detector_reference(features_df, predictions_series, latents_df)
    # Log our reference dataset
    num_iterations = 3
    for iter in range(num_iterations):
        new_features_df, new_predictions_series, new_latents_df = generate_features_labels_latents(5)
        log_new_data(detector, new_features_df, new_predictions_series, new_latents_df)
    measure = load_data()
    
    ks_results:Dict[str,any] = measure.hypothesis_tests.scipy_kolmogorov_smirnov()
    perm_results = measure.hypothesis_tests.scipy_permutation()
    log_results = measure.hypothesis_tests.logistic_detection_custom(score_type="roc_auc")

    for res in [ks_results, perm_results, log_results]:
        df = Display().table(res)
        fig, axs = Display().plot(res, score_type="statistic")


# measure = Monitor(tag="simple_example", backend=FileBackend("my-data"))
# measure.load_data()
# # print(measure.hypothesis_tests.scipy_kolmogorov_smirnov())
# # print(measure.hypothesis_tests.scipy_permutation())
# # print(measure.hypothesis_tests.scipy_mannwhitneyu())
# # print(measure.hypothesis_tests.scipy_chisquare())
# # print(measure.hypothesis_tests.gaussian_mixture_log_likelihood())
# # print(measure.hypothesis_tests.gaussian_mixture_log_likelihood(normalize=True))
# # print(measure.hypothesis_tests.logistic_detection())
# # print(measure.hypothesis_tests.logistic_detection_custom())
# # print(measure.hypothesis_tests.logistic_detection_custom(score_type="f1"))
# # print(measure.hypothesis_tests.logistic_detection_custom(score_type="roc_auc"))


# ks_results:Dict[str,any] = measure.hypothesis_tests.scipy_kolmogorov_smirnov()
# perm_results = measure.hypothesis_tests.scipy_permutation()
# log_results = measure.hypothesis_tests.logistic_detection_custom(score_type="roc_auc")

# for res in [ks_results, perm_results, log_results]:
#     df = Display().table(res)
#     fig, axs = Display().plot(res, score_type="statistic")


# print(measure.hypothesis_tests.sd_evaluate())


# logged_datasets = detector.backend.load_logged_dataset("simple_example")
# print(logged_datasets.labels)


# print(detector.registered_features)
# print(detector.registered_labels)
# print(detector.ref_dataset)
# print(measure.hypothesis_tests.kolmogorov_smirnov())
# print(measure.hypothesis_tests.sdv_evaluate())


# logged_datasets = detector.backend.load_logged_dataset("simple_example")

# print(logged_datasets.features)
# print(logged_datasets.labels)
    

if __name__ == "__main__":
    main()