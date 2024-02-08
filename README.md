# supervised-discretization

This repository contains the code for several Supervised Discretization methods. Supervised Discretization is a technique that transforms continuous features of a machine learning dataset into a set of binary features. Please note that only binary classification datasets are considered in this repository.

Here is a list of the discretization methods implemented:
* TotalDiscretizer: on each feature, a cutting point is considered between each couple of consecutive data points in the training set;
* BucketDiscretizer: on each feature, a cutting point is considered between each couple of consecutive data points in the training set with a different label;
* QuantileDiscretizer: on each feature, cutting points are generated as quantiles of the feature values; the hyperparameter **n** denotes the number of cutting points generated on each feature;
* FCCA: implements the discretization technique based on Counterfactual Analysis as presented in paper <a href="https://www.sciencedirect.com/science/article/pii/S0377221723008603">Supervised Feature Compression based on Counterfactual Analysis</a>

## Installation

* The FCCA discretizer is based on the resolution of an optimization problem, the Counterfactual Explanation problem. The MILP problem for computing the Counterfactual Explanation for a point is implemented in <a href="https://www.gurobi.com/solutions/gurobi-optimizer/?campaignid=18262689303&adgroupid=138243449982&creative=620260718865&keyword=gurobi&matchtype=e&gclid=Cj0KCQiA4OybBhCzARIsAIcfn9mYA1eyslmYMVKkmSzUWuZeLKwpNXdPrcIoKLnEr60zcnHFDSpc5j8aAgzgEALw_wcB">Gurobi</a>.
An active Gurobi Licence is needed to use FCCA.

* The package can be installed with the command:
```
pip install SupervisedDiscretization
```

## Methods
Each discretization method offers the following methods:
* **fit**: method for fitting the procedure;
* **transform**: method for discretizing a dataset by using the set of thresholds previously computed via the **fit** method;
* **fit_transform**: method for applying in sequence the **fit** and **transform** methods;

The FCCA class also offers the following method:
* **selectThresholds**: method for setting a different value of Q after the **fit** has been called; this method allows to subsample the set of thresholds in a fast way without recomputing the FCCA procedure.

## Hyperparameters
The implementation of the FCCA procedure can be found in the file *discretize.py* that contains the Python class *FCCA* which takes the following parameters:
* **estimator**: an unfitted binary classifier from the <a href='https://scikit-learn.org/stable/'>sklearn</a> package. It can be one of the following: RandomForestClassifier, GradientBoosting, LinearSVC, SVC(kernel='linear'). It is also possible to take in input GridSearchCV to choose in cross validation the parameters of the estimator;
* **p0**, **p1**: lower and upper bound for the classification probability of points for which computing the Counterfactual Explanation; 
* **lambda0**, **lambda1**, **lambda2**: hyperparameters for the Counterfactual Explanation problem that represents respectively the weights for the l0-, l1- and l2- norm;
* **compress**: boolean that is set to True to merge thresholds whose absolute difference is smaller than 0.01;
* **timelimit**: time limit in seconds for solving the Counterfactual Explanations problem.

## Execution
We report an example on how to use the FCCA procedure. The example can also be found in the file *example.py*

```
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from SupervisedDiscretization.discretizer import FCCA

if __name__ == '__main__':
    # Reading the dataset
    data = pd.read_csv('datasets/boston.csv')
    label_column = data.columns[-1]
    feature_columns = data.columns[:-1]

    # Train - test split
    data_ts = data.sample(n=int(0.3*len(data)))
    data_tr = data.drop(index=data_ts.index)

    x_tr, y_tr = data_tr[feature_columns], data_tr[label_column]
    x_ts, y_ts = data_ts[feature_columns], data_ts[label_column]

    # Target model
    target = GradientBoostingClassifier(max_depth=2, n_estimators=100,learning_rate=0.1)

    # Hyperparameters for the discretization - default values
    discretizer = FCCA(target, p0=0.5, p1=1, lambda0=0.1, lambda1=1, lambda2=0)

    # Discretization
    x_tr_discr, y_tr_discr = discretizer.fit_transform(x_tr, y_tr)
    x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts)

    # Compression - inconsistency rate
    print(f'Compression rate: {discretizer.compression_rate(x_ts, y_ts)}')
    print(f'Inconsistency rate: {discretizer.inconsistency_rate(x_ts, y_ts)}')

    print('Setting Q to 0.7')
    # Increasing the value of Q
    tao_q = discretizer.selectThresholds(0.7)

    # Discretization
    x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr, tao_q)
    x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts, tao_q)

    # Compression - inconsistency rate
    print(f'Compression rate: {discretizer.compression_rate(x_ts, y_ts, tao_q)}')
    print(f'Inconsistency rate: {discretizer.inconsistency_rate(x_ts, y_ts, tao_q)}')

```
