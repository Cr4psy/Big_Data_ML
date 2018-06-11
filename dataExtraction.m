function [] = dataExtraction(param)

%% Extraction
display 'Start data extraction'
%% VADER Dataset
if (param.includeTeacherDataset)
    display 'Importation VADER'
    filesName = {'amazonReviewSnippets_GroundTruth.txt',...
    'nytEditorialSnippets_GroundTruth.txt',...
    'movieReviewSnippets_GroundTruth.txt'};%All the files from training set 
    %Removed 'tweets_GroundTruth.txt', to use as a test dataset

%Read the data from txt file and stack them
    All = readtable(filesName{1});
    for i=2:length(filesName)
        tmpFile = readtable(filesName{i});
        if ~iscell(tmpFile.Var1)%Convert the first column into a cell
            tmpFile.Var1=num2cell(tmpFile.Var1);
        end
        All = [All; tmpFile];
    end
    All.Properties.VariableNames{3}='Data';%Rename class variable

    
    %% Extract 2/3 classes

    %Convert the linear classes into discrete classes

    %Min value:-3.87
    %Max value:3.94
    %The total range is define as +/-4, which gives for 3 classes a class
    %range of 2.67.
    
    %Class assigned values
    classNeg = -1;
    classNeu = 0;
    classPos = 1;
    
    if(param.nbClass == 2)
        %Class threshold
        classNegTh = 0;
        classPosTh = 4;   

        %Convert the classes to discrete
        %Can be optimized using matrice computation! ACL%
        All.Class = zeros(length(All.Data),1);%Initialize 0 class
        for i=1:length(All.Var2)
           if(All.Var2(i)<classNegTh)
               All.Class(i)=classNeu;
           else
               All.Class(i)=classPos;
           end
        end
    else
        %Class threshold
        classNegTh = -1.3333;
        classNeuTh = 1.3333;
        classPosTh = 4;   

        %Convert the classes to discrete
        %Can be optimized using matrice computation! ACL%
        All.Class = zeros(length(All.Data),1);%Initialize 0 class
        for i=1:length(All.Var2)
           if(All.Var2(i)<classNegTh)
               All.Class(i)=classNeg;
           elseif(All.Var2(i)<classNeuTh)
               All.Class(i)=classNeu;
           else
               All.Class(i)=classPos;
           end
        end
    end
        
end

%% KAGGLE Dataset
if (param.includeKaggleDataSet && (param.nbClass==3))
    display 'Importation KAGGLE';
    kaggle_data=load('kaggle_data_labeled.mat');

    classTempo = kaggle_data.kaggleData.Class';
    kaggleDataset=table(cell(size(classTempo)),zeros(size(classTempo)),kaggle_data.kaggleData.Data',classTempo);
    if ~exist('All','var') %If ALL doesn't exist 
        All = kaggleDataset;
        All.Properties.VariableNames{1}='Var1';
        All.Properties.VariableNames{2}='Var2';
        All.Properties.VariableNames{3}='Data';
        All.Properties.VariableNames{4}='Class';
        
    else
        %Rename class variable
        for i=1:size(All,2)
            kaggleDataset.Properties.VariableNames{i}=All.Properties.VariableNames{i};
        end
        All = [All;kaggleDataset];
    end
    
end


%% Amazon Dataset
if (param.includeAmazonDataSet && (param.nbClass==3))
    display 'Importation AMAZON';
    %amazonData=load('writtenReviewsSmall.mat');
    %amazonClass=load('ratingsSmall.mat');
    %amazon.Class = amazonClass.ratingSmall';
    %amazon.Data = amazonData.writtenReviewsSmall';
%     amazonData=load('writtenReviews.mat');
%     amazonClass=load('ratings.mat');
    tmp = load('amazonNew.mat');
    amazon.Class =  tmp.amazonNew.Class';
    amazon.Data =  tmp.amazonNew.Data';
    amazonDataset=table(cell(size(amazon.Data)),zeros(size(amazon.Data)),amazon.Data,amazon.Class);
    if ~exist('All','var') %If ALL doesn't exist 
        All = amazonDataset;
        All.Properties.VariableNames{1}='Var1';
        All.Properties.VariableNames{2}='Var2';
        All.Properties.VariableNames{3}='Data';
        All.Properties.VariableNames{4}='Class';
        
    else
        %Rename class variable
        for i=1:size(All,2)
            amazonDataset.Properties.VariableNames{i}=All.Properties.VariableNames{i};
        end
        All = [All;amazonDataset];
    end
end

    
    tabulate(All.Class)
%% Separation Training/Validation and shuffling

    display 'Shuffle data and separate in training/validation'
    %Shuffle data
    perm = randperm(length(All.Data));          %Random permutation of the data
    All.Var1 = All.Var1(perm);
    All.Var2 = All.Var2(perm);
    All.Data = All.Data(perm);
    All.Class = All.Class(perm);
    N = length(All.Data);                       %Size of the entire dataset
    
    %Cut the number of data used
    allSelected.Class = All.Class(1:round(N*param.nbDataUsed));
    allSelected.Data = All.Data(1:round(N*param.nbDataUsed));
    N = length(allSelected.Data);
    
    
    %%
    %Training set
    global training
    training.N = round(param.trainingPercentage*N);   %Size of the training dataset
    training.Class = allSelected.Class(1:training.N);   %Seperate data in two arrays
    training.Data = allSelected.Data(1:training.N);

    %Validation set
    global validation
    validation.Class = allSelected.Class(training.N+1:end);   %Seperate data in two arrays
    validation.Data = allSelected.Data(training.N+1:end);
    validation.N = length(validation.Data);           %Size of validation dataset

    
%% Test dataset
    %Use of 'tweets_GroundTruth.txt', for the test dataset
    display 'Test dataset'
    clear All
    filesName = {'tweets_GroundTruth.txt'};%All the files from training set 
    %Removed 'tweets_GroundTruth.txt', to use as a test dataset
    
    %Read the data from txt file and stack them
    All = readtable(filesName{1});

    All.Properties.VariableNames{3}='Data';%Rename class variable

    %Class assigned values
    classNeg = -1;
    classNeu = 0;
    classPos = 1;
    
    %Class threshold
    classNegTh = -1.3333;
    classNeuTh = 1.3333;
    classPosTh = 4;   

    %Convert the classes to discrete
    %Can be optimized using matrice computation! ACL%
    All.Class = zeros(length(All.Data),1);%Initialize 0 class
    for i=1:length(All.Var2)
       if(All.Var2(i)<classNegTh)
           All.Class(i)=classNeg;
       elseif(All.Var2(i)<classNeuTh)
           All.Class(i)=classNeu;
       else
           All.Class(i)=classPos;
       end
    end
    
    global test
    test.Class = All.Class;
    test.Data = All.Data;
    test.N = length(All.Class);
    

end

