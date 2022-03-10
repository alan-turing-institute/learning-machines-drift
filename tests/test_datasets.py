"""Demo datasets for developing package"""

from learning_machines_drift import datasets


def test_logistic_model() -> None:
    """Test a simple generative classification model"""

    # Given the size sample we want (N)
    N = 10

    # When we draw from a model
    X, Y = datasets.logistic_model(size=N)

    # Then we get N features and N labels back
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
