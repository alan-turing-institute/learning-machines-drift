import pandas as pd
from sklearn import datasets

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_frame["target"] = iris.target

iris_data_drift_report = Dashboard(tabs=[DataDriftTab()])
iris_data_drift_report.calculate(
    iris_frame[:100], iris_frame[100:], column_mapping=None
)
iris_data_drift_report.save("reports/my_report.html")

iris_data_and_target_drift_report = Dashboard(
    tabs=[DataDriftTab(), CatTargetDriftTab()]
)
iris_data_and_target_drift_report.calculate(
    iris_frame[:100], iris_frame[100:], column_mapping=None
)
iris_data_and_target_drift_report.save("reports/my_report_with_2_tabs.html")
