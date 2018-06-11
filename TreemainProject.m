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
param.includeTeacherDataset = true;  %Select teacher dataset
param.includeAmazonDataSet = false;     %Select amazon dataset
param.nbClass = 3;                      %Define the number of class extracted from the data set (2 or 3);
param.nbDataUsed = 0.3;                % Percentage of the total amount of the dataset selected
param.trainingPercentage = 0.7;      % Percentage of the data set used for training

%Featurization param
param.paramNbHeaders = 200;       % Minimum number of times that a word must appear to be selected in the dictionnary
param.stemming = 0;             % 1: Use stemming                  0: Doesn't use stemming
param.rmStop = 0;               % 1: Remove stops word             0: Doesn't remove it
param.bigram = 0;               % 1: Use featurize_bigram          0: Use featurize
param.crossval = 0;             % 1: Use 10-fold cross-validation  0: Don't use cross-validation

%% Global variables
global training
global validation
global test

%% Extraction
%Return the training/validation data in two global variables
dataExtraction(param);%Function to extract the data
    
%% Process

global X
global Y
extractFeatures = true; %Decide if the feature must be extracted again or if it can use the previous one

% No iterations
[accuracySummary,heads] = projectTree(param, extractFeatures);

% % Iteration (to optimize different parameters)
% param_1 = [0];                     % It can be either a value or an array for bigram
% param_2 = [0];                   % Array that turns stemming on and off
% param_3 = [1];                   % Array that turns rmStop on and of
% param_4 = [8];                   % Array for  paramNbHeaders
% length_1 = length(param_1);
% length_2 = length(param_2);
% length_4 = length(param_4);
% 
% iter = 0;                              % iterations
% accuracy=0;
% precisionPos=0;
% recallPos=0;
% precisionNeg=0;
% recallNeg=0;
% 
% for h = 1:length_1
%     bigram = param_1(h);              % Assign the parameter to param_1
%     
%     for i = 1:length_2
%        stemming = param_2(i);         % Assign the parameter to param_2
%        rmStop = param_3(i);           % Assign the parameter to param_3
%         
%         for j = 1:length_4
%             paramNbHeaders = param_4(j);   % Assign the parameter to param_4
%             iter = iter + 1;
%             [accuracy(iter),precisionPos(iter),recallPos(iter),precisionNeg(iter),recallNeg(iter),labelTest] =...
%             lab2(trainingPercentage,nbDataUsed,paramNbHeaders,stemming,rmStop,bigram,crossval);
%             labelTestMatrix(:,iter) = labelTest;
%             
%         end
%         
%     %end
% 
%     end
%     
% end 

save('headsSVM.mat','heads')

% Display the accuracy of the validation set
display('Accuracy validation set');
accuracySummary.accuracy
