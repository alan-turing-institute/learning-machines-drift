from whylogs import get_or_create_session
import pandas as pd
from sklearn import datasets


iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)

print(iris_frame)
session = get_or_create_session()


with session.logger(dataset_name="my_dataset") as logger:

    # dataframe
    logger.log_dataframe(iris_frame)

    # dict
    logger.log({"name": 1})

from whylogs.viz import profile_viewer

profile_viewer()
