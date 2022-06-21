"""Demo datasets for developing package"""

from learning_machines_drift import datasets


def test_logistic_model() -> None:
    """Test a simple generative classification model"""

    # Given the size sample we want (n)
    n_count = 10

    # When we draw from a model
    x_coord, y_coord = datasets.logistic_model(size=n_count)

    # Then we get n features and n labels back
    assert x_coord.shape[0] == 10
    assert y_coord.shape[0] == 10
