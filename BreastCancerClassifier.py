#Loading the data
from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()

#A quick look at the first observation,variable name and description suggests
#these numbers represent the radius (mean of distances from center to points
#on the perimeter)
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.DESCR)

#The response/target variable we are predicting is whether a diagnosis is
#malignant or benign.
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)
#(i.eThe first observation is a 0, indicating a malignant diagnosis)

#Splitting the data into training and validation sets
from sklearn.model_selection import train_test_split

train_test_split(
    breast_cancer_data.data, breast_cancer_data.target,
    train_size=.8, random_state=100)

#Storing each result to their respective labels
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, breast_cancer_data.target,
    train_size=.8, random_state=100)

len(training_data)
len(training_labels)
#Both sets appear to be of equal length(455 observations) indicating
#train_test_split worked successfully.

#Running the classifier
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(training_data,training_labels)

print(classifier.score(validation_data,validation_labels))
#Classifier's shows an accuracy score of 0.947368421053

#Using a for loop to determine optimal k
for k in list(range(1,100)):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data,training_labels)
    scores = classifier.score(validation_data,validation_labels)
    k_val = k+1
    print(scores,k_val)
#The best 'k' is a tie between 24, 25, and 57 with an accuracy
#of 0.964912281

#Graphing the results
import matplotlib.pyplot as plt

k_list = list(range(1,100))
accuracies = []

for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data,training_labels)
    scores = classifier.score(validation_data,validation_labels)
    accuracies.append(scores)

plt.plot(k_list, accuracies)
plt.show()

#Same plot but this time cleaned up with proper labels and title
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()

#18
#Even after tweaking train_test_split's random_state, I noticed relatively
#small changes to the accuracy score, remaining above 91%)
