function [ accuracySummary, network, heads, wordCount] = projectNN(param, extractFeatures )
global training
global validation
global test

%Global X and Y, if we want to try the algo without extracting the data
%over and over

global X
global Y
%% Neural Network
%% Prepare the data 

% the data are automatically separated into training/validation/test sets
dataNN.Class = [training.Class;validation.Class];
dataNN.Data = [training.Data;validation.Data];

%% Prepare the data
%X: Features (features:nbData)
%Y: Classes (3:nbData)
if (extractFeatures)%Can be set to false if some trials must be carried out on the neural network
display 'start feature extraction'
    % Classes    
    Y = [[dataNN.Class == -1],[dataNN.Class == 0],[dataNN.Class == 1]];
    Y = Y';%Inverse to fit with what NN expect
% Extract features
    if (param.bigram == 0)
        [wordCount, heads] = featurize(dataNN.Data,param.paramNbHeaders,param.rmStop,param.stemming);
    else
        [wordCount, heads] = featurize_bigram(dataNN.Data,param.paramNbHeaders,param.rmStop,param.stemming);
    end
    save('wordCountSaving.mat','wordCount')
    save('heads.mat','heads')
    fprintf('Number of headers: %f \n', size(wordCount,2));
    X = wordCount';
end

%% Build the neural network
display ' train the network'
net = patternnet(param.NNArchitecture,'trainscg','crossentropy');
%view(net)

%%  Train the neural network
[net,tr] = train(net,X,Y);
nntraintool

%% Validate
    testX = X(:,tr.testInd);
    testY = Y(:,tr.testInd);
    %Classify test set
    testYclassify = net(testX);
    testIndices = (vec2ind(testYclassify)-2);


%Confusion matrix
    plotconfusion(testY,testYclassify)
    %c: Confusion value = fraction of samples misclassified
    %cm: S-by-S confusion matrix, where cm(i,j) is the number of samples whose target is the ith class that was classified as j

    [c,cm,~,per] = confusion(testY,testYclassify);

    recall=per(:,2);
    precision=per(:,3);
    accuracy = (1-c);
    fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
    %fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

    accuracySummary.accuracy = accuracy;
    accuracySummary.recall = recall;
    accuracySummary.precision = precision;
    accuracySummary.F1 = 2*(recall.*precision)./(recall+precision)
    
    %Return the trained network
    network = net;


end

