function [accuracySummary,heads] = projectKNN(param, extractFeatures)

%% Importation

global training
global validation
global test

%Global X and Y, if we want to try the algo without extracting the data
%over and over

global X
global Y


%% Features extraction

display 'Extraction of the features'
% Extract the words and count the number in each sentence in the training
% set
%X: Features (features:nbData)
%Y: Classes (3:nbData)

if (extractFeatures)%Can be set to false if some trials must be carried out on the algorithm
% Extract features
    if (param.bigram == 0)
        [wordCount, heads] = featurize(training.Data,param.paramNbHeaders,param.rmStop,param.stemming);
    else
        [wordCount, heads] = featurize_bigram(training.Data,param.paramNbHeaders,param.rmStop,param.stemming);
    end
    fprintf('Number of headers: %f \n', size(wordCount,2)); %Number of columns of wordCount
    %X: predictors
    %Y: labels
    X = wordCount;
    Y = training.Class;
end

display 'Done extraction'

%% Train the KNN classifier

display 'KNN classifier training'

if (param.crossval==0)
    Mdl = fitcecoc(X,Y,'Coding','ternarycomplete','Learners','knn','ClassNames',[-1 0 1]); %k-nearest neighbours
else
    CVMdl = fitcecoc(X,Y,'Crossval','on','Coding','ternarycomplete','Learners','knn','ClassNames',[-1 0 1]); %Cross validation model (10-Fold)
    loss = kfoldLoss(CVMdl); %Loss cross-validated model
end

% For multiclass learning by combining binary SVM models, use fitcecoc (project).

% Options fitcecoc function:
% - 'Conding': 'onevsone', 'allpairs', 'binarycomplete', 'ternarycomplete' ...
% - 'FitPosterior': true of false
% - 'Learners': 'svm' (default), 'naivebayes', 'linear'...
% - 'Crossval': 'on', 'off'


display 'Done training'

%% Classify validation set

display 'Classification of the validation set'

wordCountVal = featurizeTest(validation.Data,heads,param.rmStop,param.stemming); %Extract features from validation set

if (param.crossval==0)
     labelVal = predict(Mdl,wordCountVal);   %Labels with normal classifier
else
    %Labels with cross-validated classifier
    labelVal = 0;
    kFold = length(CVMdl.Trained);           %Number k-cross validation
    
    %Evaluate the 10th different classifier
    
    for i = 1:kFold
        labelVal = labelVal+predict(CVMdl.Trained{i,1},wordCountVal);  
    end
    labelVal=round(labelVal/kFold);          %Voting
    %labelVal = str2double(labelVal);
end

    %labelVal = str2double(labelVal);
    accuracy = sum(labelVal==validation.Class)/length(validation.Class);
    
    agreePos = sum(labelVal+validation.Class==2);  %Both are 1
    valPos = sum(labelVal==1);
    GTPos = sum(validation.Class==1);
    precisionPos = agreePos/valPos;
    recallPos = agreePos/GTPos;

    agreeNeg = sum(labelVal+validation.Class==0);  %Both are 0
    valNeg = sum(labelVal==0);
    GTNeg = sum(validation.Class==0);
    precisionNeg = agreeNeg/valNeg;
    recallNeg = agreeNeg/GTNeg;
    
    recall = [recallPos recallNeg];
    precision = [precisionPos precisionNeg];
    
    accuracySummary.accuracy = accuracy;
    accuracySummary.recall = recall;
    accuracySummary.precision = precision;
    accuracySummary.F1 = 2*(recall.*precision)./(recall+precision);
        
display 'Done validation'

% %% Classify test set
% 
% display 'Classification of the test set'
% wordCountTest = featurizeTest(test.Data,heads,rmStop,stemming);
% 
% if (crossval==0)
%     labelTest = predict(Mdl,wordCountTest); 
% else
%     %Labels with cross-validated classifier
%     labelTest = 0;
%     kFold = length(CVMdl.Trained);     %Number k-cross validation
%     
%     %Evaluate the 10th different classifier
%     
%     for i = 1:kFold
%         labelTest = labelTest+predict(CVMdl.Trained{i,1},wordCountTest);  
%     end
%     labelTest=round(labelTest/kFold);  %Voting
% end
% 
% %labelVal = str2double(labelVal);
% 
% accuracy = sum(labelVal==validation.Class)/length(validation.Class);
% 
% agreePos = sum(labelVal+validation.Class==2);  %Both are 1
% valPos = sum(labelVal==1);
% GTPos = sum(validation.Class==1);
% precisionPos = agreePos/valPos;
% recallPos = agreePos/GTPos;
% 
% agreeNeg = sum(labelVal+validation.Class==0);  %Both are 0
% valNeg = sum(labelVal==0);
% GTNeg = sum(validation.Class==0);
% precisionNeg = agreeNeg/valNeg;
% recallNeg = agreeNeg/GTNeg;
% 
% accuracySummary.accuracy = accuracy;
% accuracySummary.recall = recall;
% accuracySummary.precision = precision;
% accuracySummary.F1 = 2*(recall*precision)/(recall+precision);
% 
% display 'Done classification test set'