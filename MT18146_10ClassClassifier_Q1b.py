import utils.mnist_reader
import matplotlib.pyplot as plt
import math
import numpy
import copy

def binarize(dataset):
    binarized_dataset=[]
    for i in range(0,len(dataset)):
        data = dataset[i]
        binary_data=[]
        for j in range(0,len(data)):
            if data[j]<80:
                binary_data.append(0)
            else:
                binary_data.append(1)
        binarized_dataset.append(binary_data)

    return binarized_dataset

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

def calculate_rank(x,label):
            return 10-x.index(label)

def calculate_CMC_points(rank):
    x_points=[]
    y_points=[]
    cumulative_rank=0
    for i in range(1,11):
        cumulative_rank+=rank.count(i)
        x_points.append(i)
        y_points.append(cumulative_rank)
        #print(cumulative_rank)
    return x_points,y_points

X_train, y_train = utils.mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = utils.mnist_reader.load_mnist('data/fashion', kind='t10k')

confusion_matrix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
trouser_dataset=[]
pullover_dataset=[]
tshirt_dataset=[]
dress_dataset=[]
coat_dataset=[]
sandal_dataset=[]
shirt_dataset=[]
sneaker_dataset=[]
bag_dataset=[]
ankleboot_dataset=[]
for i in range(0,len(y_train)):
    if y_train[i]==1:
        trouser_dataset.append(X_train[i])
    if y_train[i]==2:
        pullover_dataset.append(X_train[i])
    if y_train[i]==0:
        tshirt_dataset.append(X_train[i])
    if y_train[i]==3:
        dress_dataset.append(X_train[i])
    if y_train[i]==4:
        coat_dataset.append(X_train[i])
    if y_train[i]==5:
        sandal_dataset.append(X_train[i])
    if y_train[i]==6:
        shirt_dataset.append(X_train[i])
    if y_train[i]==7:
        sneaker_dataset.append(X_train[i])
    if y_train[i]==8:
        bag_dataset.append(X_train[i])
    if y_train[i]==9:
        ankleboot_dataset.append(X_train[i])

actual_test_trouser=0
actual_test_pullover=0
actual_test_tshirt=0
actual_test_dress=0
actual_test_coat=0
actual_test_sandal=0
actual_test_shirt=0
actual_test_sneaker=0
actual_test_bag=0
actual_test_ankleboot=0
for i in range(0,len(y_test)):
    if y_test[i]==1:
        actual_test_trouser+=1
    if y_test[i]==2:
        actual_test_pullover+=1
    if y_test[i]==0:
        actual_test_tshirt+=1
    if y_test[i]==3:
        actual_test_dress+=1
    if y_test[i]==4:
        actual_test_coat+=1
    if y_test[i]==5:
        actual_test_sandal+=1
    if y_test[i]==6:
        actual_test_shirt+=1
    if y_test[i]==7:
        actual_test_sneaker+=1
    if y_test[i]==8:
        actual_test_bag+=1
    if y_test[i]==9:
        actual_test_ankleboot+=1

binarized_trouser_dataset = binarize(trouser_dataset)
binarized_pullover_dataset = binarize(pullover_dataset)
binarized_tshirt_dataset = binarize(tshirt_dataset)
binarized_dress_dataset = binarize(dress_dataset)
binarized_coat_dataset = binarize(coat_dataset)
binarized_sandal_dataset = binarize(sandal_dataset)
binarized_shirt_dataset = binarize(shirt_dataset)
binarized_sneaker_dataset = binarize(sneaker_dataset)
binarized_bag_dataset = binarize(bag_dataset)
binarized_ankleboot_dataset = binarize(ankleboot_dataset)
binarized_test_dataset = binarize(X_test)
print(len(binarized_test_dataset),len(y_test))
mean_trouser = numpy.mean(binarized_trouser_dataset,axis=0)
mean_pullover = numpy.mean(binarized_pullover_dataset,axis=0)
mean_tshirt = numpy.mean(binarized_tshirt_dataset,axis=0)
mean_dress = numpy.mean(binarized_dress_dataset,axis=0)
mean_coat = numpy.mean(binarized_coat_dataset,axis=0)
mean_sandal = numpy.mean(binarized_sandal_dataset,axis=0)
mean_shirt = numpy.mean(binarized_shirt_dataset,axis=0)
mean_sneaker = numpy.mean(binarized_sneaker_dataset,axis=0)
mean_bag = numpy.mean(binarized_bag_dataset,axis=0)
mean_ankleboot = numpy.mean(binarized_ankleboot_dataset,axis=0)

variance_trouser = numpy.var(binarized_trouser_dataset,axis=0)
variance_pullover = numpy.var(binarized_pullover_dataset,axis=0)
variance_tshirt = numpy.var(binarized_tshirt_dataset,axis=0)
variance_dress = numpy.var(binarized_dress_dataset,axis=0)
variance_coat = numpy.var(binarized_coat_dataset,axis=0)
variance_sandal = numpy.var(binarized_sandal_dataset,axis=0)
variance_shirt = numpy.var(binarized_shirt_dataset,axis=0)
variance_sneaker = numpy.var(binarized_sneaker_dataset,axis=0)
variance_bag = numpy.var(binarized_bag_dataset,axis=0)
variance_ankleboot = numpy.var(binarized_ankleboot_dataset,axis=0)

true=0
rank=[]
posterior_tshirt=[]
posterior_trouser=[]
posterior_pullover=[]
posterior_dress=[]
posterior_coat=[]
posterior_sandal=[]
posterior_shirt=[]
posterior_sneaker=[]
posterior_bag=[]
posterior_ankleboot=[]

for i in range(0,len(binarized_test_dataset)):
    feature = binarized_test_dataset[i]
    label_list=[0,1,2,3,4,5,6,7,8,9]
    posterior_list=[]
    likelihood_tshirt = calculate_likelihood(mean_tshirt,feature,variance_tshirt)
    posterior_probability_tshirt = calculate_posterior_probability(likelihood_tshirt,actual_test_tshirt/len(X_test))
    posterior_list.append(posterior_probability_tshirt)
    posterior_tshirt.append(posterior_probability_tshirt)

    likelihood_trouser = calculate_likelihood(mean_trouser,feature,variance_trouser)
    posterior_probability_trouser = calculate_posterior_probability(likelihood_trouser,actual_test_trouser/len(X_test))
    posterior_list.append(posterior_probability_trouser)
    posterior_trouser.append(posterior_probability_trouser)

    likelihood_pullover = calculate_likelihood(mean_pullover,feature,variance_pullover)
    posterior_probability_pullover = calculate_posterior_probability(likelihood_pullover,actual_test_pullover/len(X_test))
    posterior_list.append(posterior_probability_pullover)
    posterior_pullover.append(posterior_probability_pullover)

    likelihood_dress = calculate_likelihood(mean_dress,feature,variance_dress)
    posterior_probability_dress = calculate_posterior_probability(likelihood_dress,actual_test_dress/len(X_test))
    posterior_list.append(posterior_probability_dress)
    posterior_dress.append(posterior_probability_dress)

    likelihood_coat = calculate_likelihood(mean_coat,feature,variance_coat)
    posterior_probability_coat = calculate_posterior_probability(likelihood_coat,actual_test_coat/len(X_test))
    posterior_list.append(posterior_probability_coat)
    posterior_coat.append(posterior_probability_coat)

    likelihood_sandal = calculate_likelihood(mean_sandal,feature,variance_sandal)
    posterior_probability_sandal = calculate_posterior_probability(likelihood_sandal,actual_test_sandal/len(X_test))
    posterior_list.append(posterior_probability_sandal)
    posterior_sandal.append(posterior_probability_sandal)

    likelihood_shirt = calculate_likelihood(mean_shirt,feature,variance_shirt)
    posterior_probability_shirt = calculate_posterior_probability(likelihood_shirt,actual_test_shirt/len(X_test))
    posterior_list.append(posterior_probability_shirt)
    posterior_shirt.append(posterior_probability_shirt)

    likelihood_sneaker = calculate_likelihood(mean_sneaker,feature,variance_sneaker)
    posterior_probability_sneaker = calculate_posterior_probability(likelihood_sneaker,actual_test_sneaker/len(X_test))
    posterior_list.append(posterior_probability_sneaker)
    posterior_sneaker.append(posterior_probability_sneaker)

    likelihood_bag = calculate_likelihood(mean_bag,feature,variance_bag)
    posterior_probability_bag = calculate_posterior_probability(likelihood_bag,actual_test_bag/len(X_test))
    posterior_list.append(posterior_probability_bag)
    posterior_bag.append(posterior_probability_bag)

    likelihood_ankleboot = calculate_likelihood(mean_ankleboot,feature,variance_ankleboot)
    posterior_probability_ankleboot = calculate_posterior_probability(likelihood_ankleboot,actual_test_ankleboot/len(X_test))
    posterior_list.append(posterior_probability_ankleboot)
    posterior_ankleboot.append(posterior_probability_ankleboot)

    label = numpy.argmax(posterior_list)

    ##writing the posterior values into the files#####################
    '''file1 = open("f1.txt","w")
    file1.write("x_axis1 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_tshirt)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis2 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_trouser)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis3 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_pullover)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis4 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_dress)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis5 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_coat)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis6 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_sandal)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis7 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_shirt)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis8 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_sneaker)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis9 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_bag)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.write("x_axis10 :\n")
    str1 = ''.join(str(e)+',' for e in posterior_ankleboot)
    str1 = '['+str1+']'
    file1.write(str1)
    file1.close()
    #print(posterior_list)'''

    #####################################################################
    if label==y_test[i]:
        true+=1
    z=[]
    confusion_matrix[y_test[i]][label]+=1
    #print(posterior_list)
    z = [x for _,x in sorted(zip(posterior_list,label_list))]
    #print(z)
    #print(y_test[i])
    r=calculate_rank(z,y_test[i])
    #print(r)
    rank.append(r)
    #if i==3:
    #    break;

print("accuracy : ",true/len(y_test))
print(confusion_matrix)
#print(rank)
sum_row=numpy.sum(confusion_matrix,axis=1)
sum_column=numpy.sum(confusion_matrix,axis=0)
for i in range(0,10):
    precision=confusion_matrix[i][i]/sum_column[i]
    print("precision: ",precision)
    recall=confusion_matrix[i][i]/sum_row[i]
    print("recall :",recall)
X_CMC_points,Y_CMC_points = calculate_CMC_points(rank)
plt.plot(X_CMC_points,Y_CMC_points)
plt.xlabel("rank")
plt.ylabel("recognition rate")
plt.show()
