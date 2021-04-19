<!-- wp:heading -->
<h2>High School Math Performance Prediction (Classification)</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><strong>Introduction</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In the world of math education, one of the major issues that
universities and educators have is that students do not succeed at mathematics
at a satisfactory level and at a rate that is satisfactory.&nbsp; Universities and educators complain of the
high failure, drop, and withdrawal rates of their students.&nbsp; This is a problem for students because low
performance in math prevents them from pursuing their degrees and careers.&nbsp; It is a problem for universities and
educators because it means that the university or educator is not successfully
teaching students, not retaining their students, and not satisfying the needs
of their students—these problems hurt the profitability and attractiveness of
the university and educator.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>If we can gain some insights into what factors most
contribute to or hurt student performance in math, we have the potential to
solve the above-mentioned problems.&nbsp; If
we can produce predictive models that can predict whether a student will pass
or fail, that can predict the numerical score of students on math assessments,
and that can predict the overall strength and promise of a student, then
universities and educators will be able to use these models to better place
students at the appropriate level of competence, to better select students for
admission, and to better understand the factors that can be improved upon to
help students be successful.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this paper, we will perform data science and machine
learning to a dataset representing the math performance of students from two
Portuguese high schools.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In a previous article, which can be found at <a href="https://medium.com/analytics-vidhya/high-school-math-performance-523d5839d7d7">High School Math Performance Regression</a>, I applied regression methods to the dataset to predict the value of G3.&nbsp; In the present paper, I would like to separate the G3 scores into five classes and try to classify a student as falling into one of five classes depending on their G3 score.&nbsp; This becomes a 5-class classification problem, and we can apply machine learning classification methods to this problem.&nbsp;</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Data
Preparation</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The data file was separated by semicolons rather than
commas.&nbsp; I replaced the semicolons by
commas.&nbsp; Then, copy and pasted everything
into notepad.&nbsp; Then, convert to a csv
file using the steps from the following link:</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://knowledgebase.constantcontact.com/articles/KnowledgeBase/6269-convert-a-text-file-to-an-excel-file?lang=en_US">https://knowledgebase.constantcontact.com/articles/KnowledgeBase/6269-convert-a-text-file-to-an-excel-file?lang=en_US</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Now, I have a nice csv file.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>There are 30 attributes that include things like student
age, parent’s education, parent’s job, weekly study time, number of absences,
number of past class failures, etc.&nbsp;
There are grades for years 1, 2, and 3; these are denoted by G1, G2, and
G3.&nbsp; The grades range from 0-20.&nbsp; G1 and G2 can be used as input features, and
G3 will be the main target output.&nbsp; </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Some of the attributes are ordinal, some are binary yes-no,
some are numeric, and some are nominal.&nbsp;
We do need to do some data preprocessing.&nbsp; For the binary yes-no attributes, I will
encode them using 0’s and 1’s.&nbsp; I did
this for schoolsup, famsup, paid, activities, nursery, higher, internet, and
romantic.&nbsp; The attributes famrel,
freetime, goout, Dalc, Walc, and health are ordinal; the values for these range
from 1 to 5.&nbsp; The attributes Medu, Fedu,
traveltime, studytime, failures are also ordinal; the values range from 0 to 4
or 1 to 4.&nbsp; The attribute absences is a
count attribute; the values range from 0 to 93.&nbsp;
The attributes sex, school, address, Pstatus, Mjob, Fjob, guardian,
famsize, reason are nominal.&nbsp; For nominal
attributes, we can use one-hot encoding.&nbsp;
The attributes age, G1, G2, and G3 can be thought of as interval attributes.

I one-hot encoded each nominal attribute, one at
a time.&nbsp; I exported the dataframe as a
csv file each time, relabeling the columns as I go.&nbsp; Finally, I reordered the columns.



</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code: </p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>import numpy as np
import pandas as pd
dataset = pd.read_csv('C:\\Users\\ricky\\Downloads\\studentmath.csv') 
X = dataset.iloc&#91;:,:-1].values
Y = dataset.iloc&#91;:,32].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Encoding binary yes-no attributes
X&#91;:,15] = labelencoder_X.fit_transform(X&#91;:,15])
X&#91;:,16] = labelencoder_X.fit_transform(X&#91;:,16])
X&#91;:,17] = labelencoder_X.fit_transform(X&#91;:,17])
X&#91;:,18] = labelencoder_X.fit_transform(X&#91;:,18])
X&#91;:,19] = labelencoder_X.fit_transform(X&#91;:,19])
X&#91;:,20] = labelencoder_X.fit_transform(X&#91;:,20])
X&#91;:,21] = labelencoder_X.fit_transform(X&#91;:,21])
X&#91;:,22] = labelencoder_X.fit_transform(X&#91;:,22])
# Encoding nominal attributes
X&#91;:,0] = labelencoder_X.fit_transform(X&#91;:,0])
X&#91;:,1] = labelencoder_X.fit_transform(X&#91;:,1])
X&#91;:,3] = labelencoder_X.fit_transform(X&#91;:,3])
X&#91;:,4] = labelencoder_X.fit_transform(X&#91;:,4])
X&#91;:,5] = labelencoder_X.fit_transform(X&#91;:,5])
X&#91;:,8] = labelencoder_X.fit_transform(X&#91;:,8])
X&#91;:,9] = labelencoder_X.fit_transform(X&#91;:,9])
X&#91;:,10] = labelencoder_X.fit_transform(X&#91;:,10])
X&#91;:,11] = labelencoder_X.fit_transform(X&#91;:,11])
onehotencoder = OneHotEncoder(categorical_features = &#91;0])
X = onehotencoder.fit_transform(X).toarray()
from pandas import DataFrame
df = DataFrame(X)
export_csv = df.to_csv (r'C:\Users\Ricky\Downloads\highschoolmath.csv', index = None, header=True)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>At this point, the final column of our dataset consists of
integers for G3.&nbsp; Scores 16-20 will form
class 1, scores 14-15 will form class 2, scores 12-13 will form class 3, scores
10-11 will form class 4, and scores 0-9 will form class 5.&nbsp; We can create a final column of classes 1-5
by converting each score to one of the classes.&nbsp;
</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code for doing it:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Defining a function that converts G3 to one of the five classes
def filter_class(score):
    if score&lt;10:
        return 5
    elif score&lt;12:
        return 4
    elif score&lt;14:
        return 3
    elif score&lt;16:
        return 2
    else:
        return 1

#defining a new column called 'class' and dropping column 'G3'
dataset_trap&#91;'class'] = dataset_trap&#91;'G3'].apply(filter_class)
dataset_trap = dataset_trap.drop(&#91;'G3'], axis=1)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Now, our dataset is ready for us to apply classification
methods on it.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Logistic
Regression</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>We define X and Y using dataset_trap.&nbsp; Then, we split the dataset into a training
set and a test set, apply feature scaling to X_train and X_test, fit logistic
regression to the training set, and predict the test set results.&nbsp; </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Define X and Y using dataset_trap
X = dataset_trap.iloc&#91;:,:-1].values
Y = dataset_trap.iloc&#91;:,-1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

#Predicting the Test set results
Y_pred = classifier.predict(X_test)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Now, we have the predicted Y values for the test set Y values.&nbsp; We can see how accurate our model is by looking at the confusion matrix:</p>
<!-- /wp:paragraph -->

<!-- wp:image {"id":397,"sizeSlug":"large"} -->
<figure class="wp-block-image size-large"><img src="https://www.onlinemathtraining.com/wp-content/uploads/2020/01/0.png" alt="" class="wp-image-397"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>The numbers in the diagonal of the confusion matrix count
the number of correct classifications.&nbsp;
So, to find how accurate our model is, we would add the diagonal entries
and divide by the total number of test set results, which is 79.&nbsp; </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code for creating the confusion matrix:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
cm.trace()/79</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>The accuracy of the model can be measured by the number of
correct predictions divided by the total number of test set results.&nbsp; In this case, the accuracy is 50%.&nbsp; This is not very impressive.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>K Nearest
Neighbors</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The k-nearest neighbors model was trained on our training set using the Euclidean distance and k=5 neighbors.&nbsp; The python code is pretty much the same as the one for logistic regression except we replace the logistic regression model with the k nearest neighbors model.&nbsp; Here is the full python code:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("studentmathdummified.csv")

#Avoiding the dummy variable trap
#Dropping GP, Male, urban,LE3, Apart,mother_at_home, father_at_home, reason_course, guardian_other
dataset_trap = dataset.drop(dataset.columns&#91;&#91;0,2,4,6,8,10,15,20,26]],axis=1)

#Defining a function that converts G3 to one of the five classes
def filter_class(score):
    if score&lt;10:
        return 5
    elif score&lt;12:
        return 4
    elif score&lt;14:
        return 3
    elif score&lt;16:
        return 2
    else:
        return 1

#defining a new column called 'class' and dropping column 'G3'
dataset_trap&#91;'class'] = dataset_trap&#91;'G3'].apply(filter_class)
dataset_trap = dataset_trap.drop(&#91;'G3'], axis=1)

#Define X and Y using dataset_trap
X = dataset_trap.iloc&#91;:,:-1].values
Y = dataset_trap.iloc&#91;:,-1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting K nearest neighbors to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)
classifier.fit(X_train,Y_train)

#Predicting the Test set results
Y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
cm.trace()/79</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>The accuracy of our model is 28%, which is really bad.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Support
Vector Machines</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Support vector machines are designed to perform classification when there are only two classes.&nbsp; However, there is a way to use svm’s when there are more than 2 classes, as in our case with five classes.&nbsp; One way is to use something called one-versus-one scheme.&nbsp; In the one-versus-one scheme, we build an svm model for every possible pair of classes.&nbsp; For K-class classification, there are K choose 2 pairs.&nbsp; So, in our case, there are 10 pairs of classes.&nbsp; We can apply build 10 support vector classifiers and predict the class for a given test point by applying all 10 support vector classifiers to the test point and choosing the class for which the number of times the test point is classified as that class is highest.&nbsp; The python code is the same as for previous classifiers except we replace the classifier with the support vector classifier:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting support vector classifier to the Training set using one-versus-one
from sklearn.svm import SVC
classifier = SVC(kernel='linear')
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>The accuracy of our model is 62%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Another way to use svm’s when there are more than 2 classes is to use one-versus-all scheme.&nbsp; In this scheme, K models are built by pairing each class with the rest.&nbsp; In our case 5 models are built.&nbsp; To predict the class for a given test point, we apply all 5 models to the test point and choose the class for which the perpendicular distance of the test point from the maximal margin hyperplane of the class’s corresponding model is largest.&nbsp; In other words, we choose the class for which the corresponding model most confidently classifies the test point as of that class.&nbsp; Here’s the python code:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting support vector classifier to the Training set using one-versus-rest
from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>The accuracy of our model is 56%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>I tried using the rbf kernel and got 44% accuracy.&nbsp; </p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting support vector classifier to the Training set using one-versus-one and rbf kernel
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>The default regularization parameter C is 1.&nbsp; I raised the regularization parameter to 4 and got an accuracy of 45.5%.&nbsp; Raising the regularization parameter to 10 gives an accuracy of 48%.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting support vector classifier to the Training set using one-versus-one and rbf kernel and C=10
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', C=10, random_state=0)
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Increasing the value of C means we’re being more strict
about how many points are misclassified; it corresponds to having a
smaller-margin separating hyperplane.&nbsp;
Put another way, by increasing C, we’re decreasing the leeway for
violations of the margin.&nbsp; Lowering the
value of C corresponds to being more lenient with misclassifications; it
corresponds to having a larger-margin separating hyperplane.&nbsp; As we increased the value of C, we saw that
the accuracy went up.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>I also tried the sigmoid kernel, with C=30, and got a 62% accuracy.&nbsp; Lowering the value of C to less than 30 or raising the value of C to higher than 30 gives poorer accuracy.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting support vector classifier to the Training set using one-versus-one and sigmoid kernel and C=30
from sklearn.svm import SVC
classifier = SVC(kernel='sigmoid', C=30, random_state=0)
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p><strong>Decision
Trees</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this method, we’re going to grow one tree.&nbsp; The splitting criterion is chosen to be entropy, and feature scaling is not necessary.&nbsp; When I applied the decision tree classifier to the test set, I got an accuracy of 72%.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting decision tree classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>I set the minimum number of samples required to split an internal node to 10 and the minimum number of samples required to be at a leaf node to 5.&nbsp; This improved the accuracy to 77%.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting decision tree classifier to the Training set with minimum number of samples required to split 10 and minimum number of leaf samples 5
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', min_samples_split=10, min_samples_leaf=5)
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p><strong>Random
Forests</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this method, we’re going to grow a bunch of trees.&nbsp; The splitting criterion is chosen to be entropy, and feature scaling is not used.&nbsp; When I applied the random forest classifier to the test set, using 10 trees, I got an accuracy of 62%:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting random forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=10)
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>I set the minimum number of samples required to split an internal node to 10 and the minimum number of samples required to be at a leaf node to 5.&nbsp; I also increased the number of trees to 100.&nbsp; This improved the accuracy to 74%:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting random forest classifier to the Training set with 100 trees and minimum samples required to split 10 and minimum samples at a leaf 5
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, min_samples_split=10, min_samples_leaf=5)
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>To improve accuracy even more, I set the max_features to None:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Fitting random forest classifier to the Training set with 100 trees, min_samples_split=10, min_samples_leaf=5, and max_features=None
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, max_features=None, min_samples_split=10, min_samples_leaf=5)
classifier.fit(X_train,Y_train)</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>I got an accuracy of 78%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Model
Selection</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In order to determine which model
is the best, we will perform k-fold cross validation (k=10) for each model and
pick the one that has the best accuracy.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For logistic regression, I got an
accuracy of 57%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For k-nearest neighbors with k=5,
I got an accuracy of 43%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For support vector classifier,
one-versus-one, I got an accuracy of 64%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For support vector classifier,
one-versus-rest, I got an accuracy of 57%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For support vector classifier,
one-versus-one with rbf kernel, I got an accuracy of 53%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For support vector classifier,
one-versus-one with rbf kernel and C=10, I got an accuracy of 57%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For support vector classifier,
one-versus-one with sigmoid kernel and C=30, I got an accuracy of 60%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For a single decision tree, I got
an accuracy of 66%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For a single decision tree with
min_samples_split=10 and min_samples_leaf=5, I got an accuracy of 69%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For random forest with 10 trees, I
got an accuracy of 64%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For random forest with 100 trees,
min_samples_split=10, and min_samples_leaf=5, I got an accuracy of 70%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For random forest with 100 trees,
min_samples_split=10, min_samples_leaf=5, max_features=None, I got an accuracy
of 76%. </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Here is the python code for applying k-fold cross validation:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train, y=Y_train, cv=10)
accuracies.mean()</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Comparing the accuracies of each model, we see that random forests with 100 trees, min_samples_split=10, min_samples_leaf=5, max_features=None has the highest accuracy.&nbsp; We might wonder whether entropy is the best criterion for splitting and whether 100 trees is the best number of trees to use; we might also wonder about what the best value for max_features is.&nbsp; I performed a grid search for criterion among entropy and gini, for n_estimators among 10,100,500, for max_features among ‘auto’, None, ‘log2’, and 1.</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>#grid search
from sklearn.model_selection import GridSearchCV
parameters = &#91;{'criterion':&#91;'entropy'],'n_estimators':&#91;10,100,500],'max_features':&#91;'auto',None,'log2',1]},{'criterion':&#91;'gini'],'n_estimators':&#91;10,100,500],'max_features':&#91;'auto',None,'log2',1]}]
grid_search=GridSearchCV(estimator = classifier, param_grid=parameters, scoring='accuracy',cv=10)
grid_search=grid_search.fit(X_train, Y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>The result is a best accuracy of 76.9% and best parameters
criterion=’gini’, max_features=None, and n_estimators=500.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>For random forest with criterion=’gini’, 500 trees,
min_samples_split=10, min_samples_leaf=5, max_features=None, I got an accuracy
of 77.5%. &nbsp;I increased min_samples_split
to 50 and got an accuracy of 78.2%.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong>Conclusion</strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In this paper, we applied logistic regression, k-nearest
neighbors, support vector classifiers, decision trees, and random forests to
the 5-class classification problem of predicting which of five classes each
student’s third year score would fall under. We found that the best performing
model, among the ones we examined, is the random forest classifier with criterion=’gini’,
500 trees, min_samples_split=50, min_samples_leaf=5, max_features=None.&nbsp; The accuracy achieved was 78.2%.&nbsp; </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>In our regression analysis of the dataset in a previous
paper, we found that some of the most significant attributes were grades in
years 1 and 2, quality of family relationships, age, and the number of
absences.&nbsp; The random forest regression
with 500 trees turned out to be one of the best performing models with 87-88%
accuracy (R squared).&nbsp; We also saw a
strong linear relationship between the grade in year 3 with the grades in years
1 and 2.&nbsp; </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Whether or not the attributes G1, G2, quality of family
relationships, age, and number of absences are always significant in every
school, in every time period, and in every country is an open question.&nbsp; Can the insights gathered here be generalized
beyond the two Portuguese high schools we considered?&nbsp; What other attributes, beside the ones we
considered, might be significant in determining math performance?&nbsp; These are open questions worth pursuing to
further understand and resolve the issue of poor math performance.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>The dataset can be found here: </p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><a href="https://archive.ics.uci.edu/ml/datasets/student+performance">https://archive.ics.uci.edu/ml/datasets/student+performance</a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.<br> <a href="http://www3.dsi.uminho.pt/pcortez/student.pdf">[Web Link]</a></p>
<!-- /wp:paragraph -->
