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
addpath ../MatlabNLP-master
addpath Dataset %Add the folder to the kernel

%% Parameters
%Data set param
param.includeKaggleDataSet = false;    %Select if we want to add the dataset from kaggle
param.includeTeacherDataset = true;  %Select teacher dataset VADER
param.includeAmazonDataSet = false;     %Select amazon dataset
param.nbClass = 3;                      %Define the number of class extracted from the data set (2 or 3);
param.nbDataUsed = 0.5;                % Percentage of the total amount of the dataset selected
param.trainingPercentage = 0.7;      % Percentage of the data set used for training

%Featurization param
param.paramNbHeaders = 65;       % Minimum number of times that a word must appear to be selected in the dictionnary
param.stemming = 1;             % 1: Use stemming                  0: Doesn't use stemming
param.rmStop = 0;               % 1: Remove stops word             0: Doesn't remove it
param.bigram = 1;               % 1: Use featurize_bigram          0: Use featurize
param.crossval = 0;             % 1: Use 10-fold cross-validation  0: Don't use cross-validation

%% Global variables
global training
global validation
global test

%% Extraction
%Return the training/validation data in two global variables
dataExtraction(param);%Function to extract the data
    
%% Process

global Y
extractFeatures = true; %Decide if the feature must be extracted again or if it can use the previous one

% % No iterations
% tic
% [accuracySummary,heads] = projectSVM(param,extractFeatures);
% toc

% Iteration (to optimize different parameters)
param_1 = [0 1];                   % It can be either a value or an array for bigram
param_2 = [0 0 1 1];                   % Array that turns stemming on and off
param_3 = [0 1 0 1];                   % Array that turns rmStop on and of
param_4 = [65];                   % Array for  paramNbHeaders
length_1 = length(param_1);
length_2 = length(param_2);
length_4 = length(param_4);

iter = 0;                              % iterations
accuracySummary.accuracy = 0;
accuracySummary.precision= [0 0];
accuracySummary.recall = [0 0];
accuracy_string = 0;

for h = 1:length_1
    param.bigram = param_1(h);              % Assign the parameter to param_1
    
    for i = 1:length_2
       param.stemming = param_2(i);         % Assign the parameter to param_2
       param.rmStop = param_3(i);           % Assign the parameter to param_3
        
        for j = 1:length_4
            param.paramNbHeaders = param_4(j);   % Assign the parameter to param_4
            iter = iter + 1;
            tic
            [accuracySummary,heads] = projectNB(param,extractFeatures);
            toc
            %extractFeatures = true; % Change to stract features only once
            accuracy_string(iter) = accuracySummary.accuracy;
            if ~exist('list','var')
                list_accuracy = {accuracySummary};
                list_heads = {heads};
            else
                list_accuracy = [list_accuracy, {accuracySummary}];
                list_heads = [list_heads, {heads}];
            end
            
        end
        
    end
    
end 

save('headsMatrix.mat','heads')

% Display the accuracy of the validation set
display('Accuracy validation set');
accuracy_string
% accuracySummary.accuracy