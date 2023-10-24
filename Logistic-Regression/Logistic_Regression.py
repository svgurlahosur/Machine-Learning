#import the essential libraries
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
#import the dataset
train_dataset = pd.read_csv('train_data.csv')
test_dataset = pd.read_csv('test_data.csv')
#copy of training dataset excluding the last column
train_input = train_dataset.iloc[:, :-1].values
#copy of last column from training dataset
train_output = train_dataset.iloc[:, -1].values
#copy of testing dataset excluding the last column
test_input = test_dataset.iloc[:, :-1].values
#copy of last column from testing dataset
test_output = test_dataset.iloc[:, -1].values

scaler = MinMaxScaler()
scaler.fit(train_input)
train_input = scaler.transform(train_input)
test_input = scaler.transform(test_input)

def hypothesis(sample, parameter):
  pred_output = parameter[0]#calculate theta0*x0 (x0 value is 1)
  for n in range(len(sample)):#calculate the number of features
    pred_output += parameter[n+1] * float((sample[n])) #calculate theta1*x1 + theta2*x2 .. . .thetan*xn
  return 1 / (1 + math.exp(-pred_output))
  
def cost(pred_output,actual_output):
  error = (pred_output - float(actual_output))
  total_cost = -(actual_output*np.log(pred_output))+(-(1-actual_output)*np.log(1-pred_output)) #total cost calculation
  return error,total_cost #error value is for parameter updation and total_cost is to measure the average error at every epoch

def gradient_descent(error,parameters,learning_rate,sample):
  parameters[0] = parameters[0] - (learning_rate * error) #update theta0
  for n in range(sample.size): #calculate the number of features
    parameters[n + 1] = parameters[n + 1] - (learning_rate * error * float(sample[n])) #update theta1, theta2, , ,, depending upon the number of features
  return parameters #return the updated paremeters

def optimization(training_input,training_output,testing_input,testing_output,learning_rate,parameters,epochs):
  best_error = 99999999999
  best_epoch = 0
  best_train_accuracy = 0
  best_test_accuracy = 0
  best_parameters = 0
  for epoch in range(epochs):
    train_cost = 0
    test_cost = 0
    for sample, actutal_output in zip(training_input,training_output):#iterate through the training dataset
      predicted_output = hypothesis(sample,parameters) #calculate the per sample predicted output
      error,total_cost = cost(predicted_output,actutal_output) #calculate error for each training sample
      parameters = gradient_descent(error,parameters,learning_rate,sample) #update the model parameters
    train_accuracy, training_cost = eval_model(training_input,training_output,parameters)
    train_cost = (training_cost/len(train_input)) #average error
    train_cost_list.append(train_cost) #append the total error from every epoch to list
    train_accuracy_list.append(train_accuracy)
    test_accuracy, testing_cost = eval_model(testing_input,testing_output,parameters) #calculate the error for every testing sample
    test_cost = (testing_cost/len(testing_input)) #sumup the error from each testing sample
    test_cost_list.append(test_cost) #append the total error from every epoch to list
    test_accuracy_list.append(test_accuracy)
    if ((epoch+1)%1000==0): #print training progress after every 1000 epochs and replace 1000 by 1 to print progress after every epoch
       print("Epoch:",epoch+1, "Training error:",train_cost, "Testing error:",test_cost,"Train accuarcy:",train_accuracy,"Testing accuarcy:",test_accuracy, "Parameters:",parameters)
    if(best_test_accuracy < test_accuracy): #copy the parameter values along with epoch number, testing error, test and train accuracy for the highest test accuracy during the training
      best_error=test_cost
      best_epoch=epoch+1
      best_parameters = parameters[:]
      best_train_accuracy = train_accuracy
      best_test_accuracy = test_accuracy
  return best_error,best_epoch,best_parameters, best_train_accuracy, best_test_accuracy

def eval_model(inputs,outputs,parameters):
  correct = 0
  total_cost = 0
  for sample, actutal_output in zip(inputs,outputs):#iterate through the dataset
    predicted_output = hypothesis(sample,parameters)#calculate the per sample predicted output
    if (round(predicted_output)==actutal_output):
      correct+=1
    error,sample_cost = cost(predicted_output,actutal_output)#calculate error for each testing sample
    total_cost += sample_cost #sumup the error from each training sample
  accuracy = (correct/len(inputs))*100
  total_cost = total_cost/len(inputs)
  return accuracy, total_cost

epochs = 10000
learning_rate = 0.0002
#calulate the number of features and initialize all the model parameters to 0
parameters = [0.0 for i in range( len(train_input[0])+1)]
#define the lists for graph plot
train_cost_list = [ ]
train_accuracy_list = [ ]
test_accuracy_list = [ ]
test_cost_list = [ ]
#start the learning algorithm
best_error,best_epoch,best_parameters,best_train_accuracy,best_test_accuracy = optimization(train_input,train_output,test_input,test_output,learning_rate,parameters,epochs)
print("\n\n\n-----------------------------------------------------Training Finished-----------------------------------------------------\n")
print("Best test accuaracy:",best_test_accuracy,"is at epoch:", best_epoch,"with train accuarcy:",best_train_accuracy,"test_error:",best_error, "and with parameters", best_parameters)
print("\n---------------------------------------------------------------------------------------------------------------------------")

#import the library to plot the graph
import matplotlib.pyplot as plt
#copy the data for plotting
epochs_num = list(range(1, epochs+1))
x = epochs_num
y1 = train_cost_list
y2 = test_cost_list
#create a figure with subplots and specify figsize
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
#plot the first subplot for training and add legend and axis labels
axes[0].plot(x, y1)
axes[0].set_title('Training error Vs Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Training error')
#plot the second subplot for testing and add legend and axis labels
axes[1].plot(x, y2)
axes[1].set_title('Testing error Vs Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Testing error')
plt.tight_layout()
#show the plots
plt.show()
plt.plot(epochs_num, train_accuracy_list, 'g', label='Training accuracy')
plt.plot(epochs_num, test_accuracy_list, 'b', label='Testing accuracy')
plt.title('Training and Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()