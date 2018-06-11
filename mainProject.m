% ------------------------------- %
% KTH Royal Institute of Technology
% Big Data in Media Technology
% Project
% Student: Anthony Clerc, Sofía Navarro Heredia, Hanna Hartikainen
% Created: September 25th, 2017
% Library from: https://github.com/faridani/MatlabNLP
% ------------------------------- %

clc
clear
close
addpath ../MatlabNLP-master%Library folder
addpath Dataset %Add the folder to the kernel

%% Parameters
    %Data set param
param.includeKaggleDataSet = true;    %Select if we want to add the dataset from kaggle
param.includeTeacherDataset = true;  %Select teacher dataset
param.includeAmazonDataSet = true;     %Select amazon dataset
param.nbClass = 3;                      %Define the number of class extracted from the data set (2 or 3);
param.nbDataUsed = 1;                % Percentage of the total amount of the dataset selected
param.trainingPercentage = 0.7;      % Percentage of the data set used for training
 %Featurization param
param.paramNbHeaders = 200;       % Minimum number of times that a word must appear to be selected in the dictionnary
param.stemming = 0;             % 1: Use stemming                  0: Doesn't use stemming
param.rmStop = 0;               % 1: Remove stops word             0: Doesn't remove it
param.bigram = 0;               % 1: Use featurize_bigram          0: Use featurize
 %Classifier SVM, NB param
param.crossval = 0;             % 1: Use 10-fold cross-validation  0: Don't use cross-validation
 %Classifier NN param
param.NNArchitecture = [10];   %Architecture of the Neural network, hidden layers and number of points
%% Global variables
global training
global validation
global test

%% Extraction
%Return the training/validation data in two global variables
tic
dataExtraction(param);%Function to extract the data
display('Extraction time')
toc

%% Neural Network
global X
global Y
extractFeatures = false; %Decide if the feature must be extracted again or if it can use the previous one
tic
[accuracySummary, network, heads] = projectNN(param, extractFeatures);
display('Feature extraction and training')
toc

%% Support Vector
%[accuracySummary, network, heads] = projectSVM(param, extractFeatures);

%% KNN
%[accuracySummary, network, heads] = projectKNN(param, extractFeatures);

%% DT
%[accuracySummary, network, heads] = projectDT(param, extractFeatures);

%% NB
%[accuracySummary, network, heads] = projectNB(param, extractFeatures);

%%
save('Experimentation/network.net','network')
%save('Experimentation/heads.mat','heads')
save('Experimentation/accuracySummary.mat','accuracySummary')


