# DATA Analysis VALVE DF31
By Ilyas Elamri (https://www.linkedin.com/in/ilyas-elamri-0548b4a3/)

<p align="center"><img src="/img/valve.png" alt="drawing" width="300" height="300"/></p>

## Introduction

The business objective of the Advanced Analytics project is to focus on the current relation between the Side streaming max test on the valve DF-31 and the existing Intermediate Critical Quality Attributes. The expected deliverable is the list of Intermediate Critical Quality Attributes directly affecting the Side streaming max with associated influence weight.

The Side streaming max is one of the most important quality attributes of the Valve DF-31 products and its variability depends on multiple causes including but not limited to the components molding and the overall process execution.

At the end of the project, Customer’s team needs to identify all necessary lever of actions in order to stay within their Specifications.

## I. Overview
python.version| 3.6 
------------------------ | -------------

> More details in requirements.


###### Deployment

- From the two large tables Genealogy.csv and Results.csv which contain the datas of all the material codes, we retrieve those that interest us, i.e. the results of all the tests carried out on the valve DF-31 as well as these components with the information on the link between the components batches and the final product batches :
```
bash filter_results.sh
bash filter_genealogy.sh
```

- From the filtred results table, we plot distribution, scatterplot and we save some statistics :
```
python get_scatterplot.py
python get_stats_distribution.py
```

- From the filtred results table, we clean data:
```
python clean_results.py
```

- From the mean results per batches and the genealogy we build our final dataset:
```
bash get_dataset.sh
```

- To get informations about all dataset columns run:
```
python describe_dataset.py
```


## II. Dataset description 

The dataset in our possession is built from the genealogy between each batch of the assembled material as well as the average per batch of tests on each component, which leads us back to describing each target variable (side streaming max) by a vector of 627 attributes.

A serious problem in mining our database is that they are incomplete, and a significant amount of data is missing, or erroneously entered. 
The graph bellow shows the initial distribution of the missing values for each variable of our dataset.

<p align="center"><img src="/img/missingValues1.png" alt="drawing"/></p>

<table align="center">
	<tr><th>Rate of missing values</th><th>Size of elements</th><th>Size of variables</th>
	<tr><td>23.03 %</td><td>335</td><td>627</td>				
</table>


## III. Preprocessing

- Remove rows (batches) that have no values for the target variable (side streaming max)
- Combination of variables that represent the same test: 
	- Example: "Diameter of HDS" and "Diameter HDS" carried out on the Valve DF31
	- Number of combined tests: 19
- Elimination of tests deemed non-impacting:
	- Number of deleted tests: 196
- Creation of a new variable from the difference of the two tests: “K PI” and “K inv PI” performed on the Stem-DF31 — PBT-NATU

Bellow the new distribution of missing values:

<p align="center"><img src="/img/missingValues2.png" alt="drawing" width="600" height="400"/></p>

<table align="center">
	<tr><th>Rate of missing values</th><th>Size of elements</th><th>Size of variables</th></tr>
	<tr><td>19.07 %</td><td>320</td><td>418</td></tr>				
</table>

- Delete variables that have more than 15% missing values.
- Delete batches that have more than 5% missing values.
- Delete tests that have zero variance.
	- Number of deleted tests: 95

<p align="center"><img src="/img/missingValues3.png" alt="drawing" width="500" height="400"/></p>

<table align="center">
	<tr><th>Rate of missing values</th><th>Size of elements</th><th>Size of variables</th></tr>
	<tr><td>0.28 %</td><td>272</td><td>239</td></tr>				
</table>

- Homoscedasticity test: Levene's test assesses whether the population variances are equal. If the p-value resulting from the Levene's test is below a level of significance (typically 0.05), it is unlikely that the differences obtained in the sample variances occurred based on a random sampling of a population. with equal variances. Thus, the null hypothesis of equal variance is rejected, and it is concluded that there is a difference between the variances in the population.
	- Number of deleted tests: 58

<p align="center"><table>
	<tr><th>Rate of missing values</th><th>Size of elements</th><th>Size of variables</th></tr>
	<tr><td>0 %</td><td>272</td><td>181</td></tr>				
</table></p>


## IV. Distribution of the side streaming max

- Number of batches with side streaming max at 0: 196
- Number of batches with a side streaming max different than 0: 76
 
<p align="center"><img src="/img/distribution.png" alt="drawing" width="600" height="250"/></p>


we binarize the values of the max side streaming in order to obtain a dataset ready for a binary classification. Bellow the distribution of the two classes of the side streaming max.
- label 0: no side streaming
- label 1: side streaming different from null

<p align="center"><img src="/img/repartition.png" alt="drawing" width="250" height="250"/></p>


## V. Data visualization

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.

<p align="center"><img src="/img/pca.png" alt="drawing" width="600" height="400"/></p>

<p align="center"><img src="/img/pca3D.png" alt="drawing" width="600" height="400"/></p>

## VI. Oversampling

A problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary.

One way to solve this problem is to oversample the examples in the minority class. This can be achieved by simply duplicating examples from the minority class in the training dataset prior to fitting a model. This can balance the class distribution but does not provide any additional information to the model.
An improvement on duplicating examples from the minority class is to synthesize new examples from the minority class. This is a type of data augmentation for tabular data and can be very effective.
Perhaps the most widely used approach to synthesizing new examples is called the Synthetic Minority Oversampling Technique or SMOTE for short.

SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.
Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen, and a synthetic example is created at a randomly selected point between the two examples in feature space.

<p align="center"><img src="/img/dataAugmentation.png" alt="drawing" width="600" height="400"/></p>


<p align="center"><img src="/img/dataAugmentation3D.png" alt="drawing" width="600" height="400"/></p>


## VII. Most impacting attributes

#### 1. Point biserial correlation coefficient

The point biserial correlation coefficient (rpb) is a correlation coefficient used when one variable (e.g. Y) is dichotomous; Y can either be "naturally" dichotomous, like whether a coin lands heads or tails, or an artificially dichotomized variable. In most situations it is not advisable to dichotomize variables artificially. When a new variable is artificially dichotomized the new dichotomous variable may be conceptualized as having an underlying continuity. If this is the case, a biserial correlation would be the more appropriate calculation.
The point-biserial correlation is mathematically equivalent to the Pearson (product moment) correlation, that is, if we have one continuously measured variable X and a dichotomous variable Y, rXY = rpb. This can be shown by assigning two distinct numerical values to the dichotomous variable.

<p align="center"><img src="/img/correlation.png" alt="drawing" width="700" height="350"/></p>

 
•	Top positive correlations:
<p align="center"><img src="/img/positiveCorrelation.png" alt="drawing" width="600" height="300"/></p>

•	Top negative correlations:
<p align="center"><img src="/img/negativeCorrelation.png" alt="drawing" width="600" height="300"/></p>

 

#### 2. Embedded Method for features selection

Embedded methods are iterative in a sense that takes care of each iteration of the model training process and carefully extract those features which contribute the most to the training for a particular iteration. Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.

Here we will do feature selection using Lasso regularization. If the feature is irrelevant, lasso penalizes its coefficient and make it 0. Hence the features with coefficient = 0 are removed and the rest are taken.

<p align="center"><img src="/img/lasso.png" alt="drawing" width="350" height="100"/></p>

#### 3. Gini importance

The importance of the characteristics is calculated as the decrease in the impurity of the node weighted by the probability of reaching this node. The probability of the node can be calculated by the number of samples reaching the node, divided by the total number of samples.

<p align="center"><img src="/img/gini.png" alt="drawing" width="600" height="250"/></p> 

#### 4. Density of each class for each selected variable 

The following table contains the selected variables to train our statistical model for prediction.

<p align="center"><img src="/img/selected.png" alt="drawing" width="600" height="200"/></p>

<div class='row'>
	<img src="/img/b11.png" alt="drawing" width="300" height="200"/>
	<img src="/img/b12.png" alt="drawing" width="300" height="200"/>
	<img src="/img/conc.png" alt="drawing" width="200" height="200"/>
	<img src="/img/diametre1.png" alt="drawing" width="300" height="200"/>
	<img src="/img/diametre2.png" alt="drawing" width="300" height="200"/>
	<img src="/img/std.png" alt="drawing" width="200" height="200"/>
</div>
<p align="center"><img src="/img/allongement.png" alt="drawing" width="430" height="200"/></p>

## VIII. Prediction

#### 1. Support Vector Machine (SVM)
SVMs are a family of machine learning algorithms that solve classification, regression, and anomaly detection problems. They are known for their solid theoretical guarantees, their great flexibility as well as their ease of use even.

SVMs were developed in the 1990s, their principle is simple: their purpose is to separate data into classes using a border as "simple" as possible, to so that the distance between the different groups of data and the border between them is maximum. This distance is also called "margin" and the SVMs are thus qualified as "wide margin separators", the "support vectors" being the data closest to the border.

###### Linear classification problem
For a linear classification problem we assume that the two classes are separable by a hyperplane, the function f therefore has the form:

<p align="center"><img src="/img/math.png" alt="drawing" width="300" height="75"/></p>

where w is the vector orthogonal to the hyperplane and b is the displacement compared to the origin.

<p align="center"><img src="/img/svm.png" alt="drawing" width="600" height="400"/></p>

To judge the quality of a hyperplane as a separator, we use the distance between the training examples and this separator. More specifically, the "margin" of a learning problem is defined as the distance between the closest learning example and the separation hyperplane. For a hyperplane H we have:

<p align="center"><img src="/img/marge.png" alt="drawing" width="200" height="50"/></p>

We thus arrive at the following optimization problem (called primal problem):

<p align="center"><img src="/img/primal.png" alt="drawing" width="300" height="85"/></p>

###### Non linear classification problem
Often it happens that even if the problem is linear, the data is affected by noise (eg from sensor) and the two classes are mixed around the separation hyperplane. To manage this type of problem we use a technique called flexible margin, which tolerates bad rankings:

- Add stress relaxation variables ξi
- Penalize these lapses in the objective function.

<p align="center"><img src="/img/separation.png" alt="drawing"/></p>

The optimization problem in the case of non-separable data is therefore:

<p align="center"><img src="/img/dual.png" alt="drawing" width="300" height="150"/></p>

C It is a variable for penalizing badly classified points making a compromise between the width of the margin and badly classified points. The variables ξi are also called spring variables


#### 2. k-fold Cross-Validation
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.

Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.
It is a popular method because it is simple to understand and because it generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.
The general procedure is as follows:
- Shuffle the dataset randomly.
- Split the dataset into k groups
- For each unique group:
	- Take the group as a hold out or test data set
	- Take the remaining groups as a training data set
	- Fit a model on the training set and evaluate it on the test set
	- Retain the evaluation score and discard the model
- Summarize the skill of the model using the sample of model evaluation scores

Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times.

#### 3. Application of model on our dataset
We fit our SVM model with a gaussian kernel, with a regularization parameter of C=1000 and a gaussian parameter gamma=2.1.
We use a 5 cross validation to analyze the performance of our model. For each fold, we calculate the score off our prediction.


#### 4. Metrics

<p align="center"><img src="/img/metrics.png" alt="drawing" width="600" height="350"/></p>

- True positive: Batches with no side streaming correctly identified as batches with no side streaming 
- False positive: Batches with side streaming incorrectly identified as batches with no side streaming 
- True negative: Batches with side streaming correctly identified as batches with side streaming 
- False negative: Batches with no side streaming incorrectly identified as batches with side streaming

#### 5. Results

<table align="center">
	<tr><th>Folds</th><th>Score</th></tr>
	<tr><td>1</td><td>86.25 %</td></tr>		
	<tr><td>2</td><td>89.87 %</td></tr>		
	<tr><td>3</td><td>94.94 %</td></tr>		
	<tr><td>4</td><td>92.40 %</td></tr>		
	<tr><td>5</td><td>91.14 %</td></tr>	
	<tr><td>Average score</td><td>90.09 %</td></tr>				
</table>


Now we have a good model (90.09 % of score), we split our dataset into training sub dataset and test sub dataset.

<table align="center">
	<tr><th>&nbsp;</th><th>SIZE OF DATA</th><th>RATE</th></tr>
	<tr><td>TRAIN</td><td>356</td><td>90 %</td></tr>		
	<tr><td>TEST</td><td>40</td><td>10 %</td></tr>					
</table>


We fit our model with the training sub dataset, and we test on the test sub dataset. We get the score bellow:
Prediction score: 93 %

#### 6. Confusion matrix
<p align="center"><img src="/img/confusion.png" alt="drawing"/></p>


#### 7. Classification report
<p align="center"><img src="/img/report.png" alt="drawing"  width="400" height="150"/></p>


#### 8. ROC curve (receiver operating characteristic)
A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The true-positive rate is also known as sensitivity, recall or probability of detection in machine learning. The false-positive rate is also known as probability of false alarm and can be calculated as (1 − specificity). It can also be thought of as a plot of the power as a function of the Type I Error of the decision rule (when the performance is calculated from just a sample of the population, it can be thought of as estimators of these quantities). The ROC curve is thus the sensitivity as a function of fall-out. In general, if the probability distributions for both detection and false alarm are known, the ROC curve can be generated by plotting the cumulative distribution function (area under the probability distribution from −∞ to the discrimination threshold) of the detection probability in the y-axis versus the cumulative distribution function of the false-alarm probability on the x-axis.

<p align="center"><img src="/img/roc.png" alt="drawing" width="300" height="400"/></p>

#### 9. Decision boundary

- Red area represents the side streaming different from 0

- Blue area represents the side streaming 0

<div class='row'>
	<img src="/img/b11_allongement.png" alt="drawing" width="400" height="350"/>
	<img src="/img/b12Allongement.png" alt="drawing" width="400" height="350"/>
	<img src="/img/conc_allongement.png" alt="drawing" width="400" height="350"/>
	<img src="/img/diametre1_allongement.png" alt="drawing" width="400" height="350"/>
	<img src="/img/diametre2_allongement.png" alt="drawing" width="400" height="350"/>
	<img src="/img/std_allongement.png" alt="drawing" width="400" height="350"/>
	<img src="/img/b11_b12.png" alt="drawing" width="400" height="350"/>
	<img src="/img/conc_b12.png" alt="drawing" width="400" height="350"/>
	<img src="/img/diametre1_b12.png" alt="drawing" width="400" height="350"/>
	<img src="/img/diametre2_b12.png" alt="drawing" width="400" height="350"/>
	<img src="/img/std_b12.png" alt="drawing" width="400" height="350"/>
	<img src="/img/conc_b11.png" alt="drawing" width="400" height="350"/>
	<img src="/img/diametre1_b11.png" alt="drawing" width="400" height="350"/>
	<img src="/img/diametre2_b11.png" alt="drawing" width="400" height="350"/>
	<img src="/img/std_b11.png" alt="drawing" width="400" height="350"/>
	<img src="/img/conc_diametre1.png" alt="drawing" width="400" height="350"/>
	<img src="/img/conc_diametre2.png" alt="drawing" width="400" height="350"/>
	<img src="/img/std_conc.png" alt="drawing" width="400" height="350"/>
	<img src="/img/diametre1_diametre2.png" alt="drawing" width="400" height="350"/>
	<img src="/img/std_diametre1.png" alt="drawing" width="400" height="350"/>
	<img src="/img/std_diametre2.png" alt="drawing" width="400" height="350"/>

</div>
