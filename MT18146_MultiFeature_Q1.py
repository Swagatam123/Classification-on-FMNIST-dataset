import utils.mnist_reader
import matplotlib.pyplot as plt
import math
import numpy
import copy

def binarize(trouser,pullover,test_dataset):
    binarized_trouser_dataset=[]
    for i in range(0,len(trouser)):
        trouser_data = trouser[i]
        binary_trouser=[]
        for j in range(0,len(trouser_data)):
            if trouser_data[j]<80:
                binary_trouser.append(0)
            else:
                binary_trouser.append(1)
        binarized_trouser_dataset.append(binary_trouser)
    binarized_pullover_dataset=[]
    for i in range(0,len(pullover)):
        pullover_data = pullover[i]
        binary_pullover=[]
        for j in range(0,len(pullover_data)):
            if pullover_data[j]<80:
                binary_pullover.append(0)
            else:
                binary_pullover.append(1)
        binarized_pullover_dataset.append(binary_pullover)
    binarized_test_dataset=[]
    for i in range(0,len(test_dataset)):
        test_data = test_dataset[i]
        binary_test_data=[]
        for j in range(0,len(test_data)):
            if test_data[j]<80:
                binary_test_data.append(0)
            else:
                binary_test_data.append(1)
        binarized_test_dataset.append(binary_test_data)

    return binarized_trouser_dataset,binarized_pullover_dataset,binarized_test_dataset

def count_ones(data_set):
    count =0
    for i in range (0,len(data_set)):
        if data_set[i]==1:
            count+=1
    return count

def calculate_likelihood(mean,feature,sigma):
    likelihood=0
    for i in range(0,len(feature)):
        if sigma[i]==0:
            sigma[i]=0.1
        temp = 1/(pow(2*3.14*sigma[i],0.5))*math.exp(-1*(pow(feature[i]-mean[i],2)/(2*sigma[i])))
        if temp!=0:
            likelihood+=math.log10(temp)
    return likelihood

def calculate_posterior_probability(likelihood,prior):
    posterior_probability = likelihood+math.log10(prior)
    return posterior_probability

def ROC_curve_points(posterior_trouser_dataset,binarized_test_dataset,label_test_dataset,actual_test_trouser,actual_test_pullover):
    posterior = copy.copy(posterior_trouser_dataset)
    posterior_trouser_dataset.sort()
    x_points=[]
    y_points=[]
    for threshold in range(0,len(posterior_trouser_dataset)):
        fp=0
        tp=0
        for i in range(0,len(binarized_test_dataset)):
            #feature = binarized_test_dataset[i]
            #likelihood_trouser = calculate_likelihood(mean_trouser,feature,variance_trouser)
            #posterior_probability_trouser = calculate_posterior_probability(likelihood_trouser,actual_test_trouser/(actual_test_pullover+actual_test_trouser))
            if posterior[i] >= posterior_trouser_dataset[threshold]:
                if label_test_dataset[i]==1:
                    tp+=1
                else:
                    fp+=1
        y_points.append(fp/actual_test_pullover)
        x_points.append(tp/actual_test_trouser)
        print(threshold,fp/actual_test_pullover,tp/actual_test_trouser)

    return x_points,y_points


X_train, y_train = utils.mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = utils.mnist_reader.load_mnist('data/fashion', kind='t10k')

trouser_dataset=[]
pullover_dataset=[]
for i in range(0,len(y_train)):
    if y_train[i]==1:
        trouser_dataset.append(X_train[i])
    if y_train[i]==2:
        pullover_dataset.append(X_train[i])

test_dataset=[]
label_test_dataset=[]
actual_test_trouser=0
actual_test_pullover=0
for i in range(0,len(y_test)):
    if y_test[i]==1 or y_test[i]==2:
        test_dataset.append(X_test[i])
        if y_test[i]==1:
            label_test_dataset.append(1)
            actual_test_trouser+=1
        elif y_test[i]==2:
            label_test_dataset.append(2)
            actual_test_pullover+=1
binarized_trouser_dataset,binarized_pullover_dataset,binarized_test_dataset = binarize(trouser_dataset,pullover_dataset,test_dataset)

#feature_vector_trouser=[]
#feature_vector_pullover=[]
mean_trouser = numpy.mean(binarized_trouser_dataset,axis=0)
mean_pullover = numpy.mean(binarized_pullover_dataset,axis=0)
variance_trouser = numpy.var(binarized_trouser_dataset,axis=0)
variance_pullover = numpy.var(binarized_pullover_dataset,axis=0)
#print(len(variance_trouser))
true=0
posterior_trouser_dataset=[]
predicted_trouser_count=0
predicted_pullover_count=0
confusion_matrix = [[0,0],[0,0]]
for i in range(0,len(binarized_test_dataset)):
    feature = binarized_test_dataset[i]
    likelihood_trouser = calculate_likelihood(mean_trouser,feature,variance_trouser)
    posterior_probability_trouser = calculate_posterior_probability(likelihood_trouser,actual_test_trouser/(actual_test_pullover+actual_test_trouser))
    likelihood_pullover = calculate_likelihood(mean_pullover,feature,variance_pullover)
    posterior_probability_pullover = calculate_posterior_probability(likelihood_pullover,actual_test_pullover/(actual_test_pullover+actual_test_trouser))
    posterior_trouser_dataset.append(posterior_probability_trouser)
    label=0
    if posterior_probability_pullover>posterior_probability_trouser:
        label=2
    else:
        label=1
    if label==label_test_dataset[i]:
      true+=1
      if label==1:
          predicted_trouser_count+=1
          confusion_matrix[0][0]+=1
      else:
          predicted_pullover_count+=1
          confusion_matrix[1][1]+=1
    elif label==1 and label_test_dataset[i]==2:
            confusion_matrix[0][1]+=1
    elif label==2 and label_test_dataset[i]==1:
            confusion_matrix[1][0]+=1

print("accuracy : ",true/len(label_test_dataset))
print("precision : ",confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]))
print("recall : ",confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0]))
x_points,y_points=ROC_curve_points(posterior_trouser_dataset,binarized_test_dataset,label_test_dataset,actual_test_trouser,actual_test_pullover)
plt.plot(y_points,x_points)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()
print(confusion_matrix)

