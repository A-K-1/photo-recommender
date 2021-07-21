# photo-recommender
Dissertation project


## Hypothesis 
A system that uses the random forest algorithm will produce more accurate and relevant predictions compared to a system that uses decision trees, an artificial neural network or a support vector machine on a small dataset (less than one hundred samples). The accuracy of each model and relevance of the predictions will be calculated and compared using leave-one-out cross validation to test all data points.
Decision trees are a very popular machine learning algorithm, one that is powerful yet easy to interpret. Artificial neural networks (ANNs) aim to simulate the interconnected brain cells of the human brain thus allowing decisions to be made in a way similar to how humans make decisions [30]. Support vector machines (SVMs) are another approach to machine learning that are effective on both linear and non-linear data.  Random forests are an ensemble method which is also a popular approach that creates multiple decision trees to find the best possible solution. Due to the popularity of all the machine learning approaches, it is important to determine which approach does in fact produce more accurate results, and in the context of this project, the most relevant predictions for each user. Random forests aim to alleviate the problem of overfitting, thus making the model more generalisable and increasing its accuracy [5][18]. This project will set out to prove that this is the case and that random forests is superior to decision trees, ANNs and SVMs in terms of performance.



## Aim
To accurately predict relevant photography locations to a user based on their preferences and tags from locations they have previously liked by training a random forest, decision tree, artificial neural network and support vector machine.



## Project Description
To determine whether the random forest machine learning algorithm will have a superior performance to an ANN, decision tree or SVM, the models will be trained on data submitted by photographers. The data includes information on what types of photography the users prefer, their three favourite photography locations and the tags extracted from images of their favourite locations. The features to learn will be the preferred photography types and photography tags associated with each user. The four algorithms will attempt to correctly predict the users’ three favourite photography locations (the labels) after being trained on the user data.	

The performance of the random forests, decision trees, ANN and SVM on the prediction of the three locations for users will be compared using leave-one-out cross validation (see section 3.8). This will allow the comparison of the accuracy of the four approaches as well as the relevance of the predictions produced by each, which will be used to determine whether random forests are superior on smaller datasets (less than one hundred samples). As random forests reduce the chance of overfitting [5][18] whilst training a large number of trees to obtain an optimal output, it is expected that they will have a better performance due to the increased generalisability of the model that is trained.

The Tanimoto Coefficient similarity metric will be used to calculate how similar the photography types are of the predicted photography location to the photography types preferred by the given user. This calculation will be used to calculate how relevant a predicted location is to the user, which is of a high importance. An incorrect prediction can still be useful to a user if they are of a high relevance to the user’s photography interests. The average relevance of all predictions for each algorithm will also be compared.
	
The system will be coded in Python and the emphasis is on the system being able to correctly predict a user’s favourite photography locations or produce predictions of a high relevance to a user. The machine learning approaches will be evaluated and the results that each produce are the focus. As a result, a front-end user interface will not be implemented as it is not required to answer the research question posed in this project.



## Conclusion and Recommendation for Further Research
‘Machine learning classification requires thorough fine tuning of the parameters and at the same time a sizeable number of instances for the data set’ [12]. In this project, a small dataset with less than one-hundred samples was used to determine which machine learning algorithm was able to produce the best performance. Performance was not based solely on the accuracy of predictions made by each algorithm, but also by how relevant each prediction was to a user (measured using the Tanimoto Coefficient – section 3.6), independent of whether the prediction was right or wrong. 

A dummy predictor was used to provide a baseline for comparison, all approaches produced a higher performance to the baseline. Under the given conditions, the random forest algorithm produced predictions with the highest accuracy, however, the SVM produced predictions with the highest relevance to a given user. Overall, the SVM had the superior performance, when taking into account both accuracy and relevance, despite this, the random forest algorithm had an overall performance which was only 2.6% worse than the SVM. Although the hypothesis has been proven false by the results, which show that the SVM has the most relevant and accurate predictions, due to the closeness of the performance values of the SVM and random forest, we can conclude that both approaches have a superior performance on a small dataset when compared to ANNs and decision trees.

In addition to the research carried out in this project, which shows that SVMs and random forests have a high performance when trained on a small dataset, further research and study could be carried out with a small dataset. The small dataset that is used for training could introduce missing data or fewer/more features to detect which machine learning algorithm can handle these issues most efficiently. It would be interesting to observe whether the SVM and random forest algorithms still maintained the best performance under these conditions.

