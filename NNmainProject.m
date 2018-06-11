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
param.nbDataUsed = 0.3;                % Percentage of the total amount of the dataset selected
param.trainingPercentage = 0.7;      % Percentage of the data set used for training
 %Featurization param
param.paramNbHeaders = 1500;       % Minimum number of times that a word must appear to be selected in the dictionnary
param.stemming = 0;             % 1: Use stemming                  0: Doesn't use stemming
param.rmStop = 0;               % 1: Remove stops word             0: Doesn't remove it
param.bigram = 0;               % 1: Use featurize_bigram          0: Use featurize
 %Classifier SVM, NB param
param.crossval = 0;             % 1: Use 10-fold cross-validation  0: Don't use cross-validation
 %Classifier NN param
param.NNArchitecture = [80,20,10];   %Architecture of the Neural network, hidden layers and number of points
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
extractFeatures = true; %Decide if the feature must be extracted again or if it can use the previous one
for i = 1:10
    param.nbDataUsed = i/10;
    tic
    [accuracySummary, network, heads, wordCount] = projectNN(param, extractFeatures);
    display('Feature extraction and training')
    toc
    filename = sprintf('%s_%d.mat','Experimentation/wordCount',i)
    save(filename,'wordCount')
   %save('Experimentation/network.net','network')
    filename = sprintf('%s_%d.net','Experimentation/network',i)
    save(filename,'network')
    %save('Experimentation/heads.mat','heads')
    filename = sprintf('%s_%d.mat','Experimentation/heads',i)
    save(filename,'heads')
    %save('Experimentation/accuracySummary.mat','accuracySummary')
    filename = sprintf('%s_%d.mat','Experimentation/accuracySummary',i)
    save(filename,'accuracySummary')
    %save('Experimentation/param.mat','param')
    filename = sprintf('%s_%d.mat','Experimentation/param',i)
    save(filename,'param')
    close
end
%% Process

% global X
% global Y
% extractFeatures = true; %Decide if the feature must be extracted again or if it can use the previous one

% % No iterations
% tic
% [accuracySummary,heads] = projectSVM(param,extractFeatures);
% toc

% Iteration (to optimize different parameters)
% param_1 = [0 1];                   % It can be either a value or an array for bigram
% param_2 = [0 1 0 1];                   % Array that turns stemming on and off
% param_3 = [0 0 1 1];                   % Array that turns rmStop on and of
% param_4 = [250];                   % Array for  paramNbHeaders
% length_1 = length(param_1);
% length_2 = length(param_2);
% length_4 = length(param_4);
% 
% iter = 0;                              % iterations
% accuracySummary.accuracy = 0;
% accuracySummary.precision= [0 0];
% accuracySummary.recall = [0 0];

% for h = 1:length_1
%     param.bigram = param_1(h);              % Assign the parameter to param_1
%     
%     for i = 1:length_2
%        param.stemming = param_2(i);         % Assign the parameter to param_2
%        param.rmStop = param_3(i);           % Assign the parameter to param_3
%         
%         for j = 1:length_4
%             param.paramNbHeaders = param_4(j);   % Assign the parameter to param_4
%             iter = iter + 1;
%             tic
%             %[accuracySummary,heads] = projectSVM(param,extractFeatures);
%             
%             [accuracySummary, network, heads] = projectNN(param, extractFeatures);
%             toc
%             %headsMatrix(:,iter) = heads;
%             if ~exist('list','var')
%                 accuracyList = {accuracySummary};
%                 networkList = {network};
%                 headsList = {heads};
%             else
%                 accuracyList = [accuracyList, {accuracySummary}];
%                 networkList = [networkList, {network}];
%                 headsList = [headsList, {heads}];
%             end
%             close;
%         end
%         
%     end
%     
% end 
% 
% save('Experimentation/accuracyList.mat','accuracyList')
% save('Experimentation/networkList.mat','networkList')
% save('Experimentation/headsList.mat','headsList')
% save('Experimentation/param.mat','param')
% 
% % Display the accuracy of the validation set
% display('Accuracy validation set');
% accuracySummary.accuracy

%%
% save('Experimentation/network.net','network')
% %save('Experimentation/heads.mat','heads')
% save('Experimentation/accuracySummary.mat','accuracySummary')

%% SVM 
% % [accuracy,precisionPos,recallPos,precisionNeg,recallNeg,labelTest] = projectSVM(param.trainingPercentage,param.nbDataUsed,param.paramNbHeaders,param.stemming,param.rmStop,param.bigram,param.crossval); 
% % display('Accuracy validation set');
% % accuracy
