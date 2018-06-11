clear
close
clc

 load('../Dataset/headsNN.mat')
%  MdlNB=load('..\Big Data\MdlNB.mat')
%  MdlKNN=load('..\Big Data\MdlKNN.mat')
%  MdlDT=load('..\Big Data\MdlDT.mat')
 MdlNN=load('../Dataset/MdlNN.mat')
%  MdlNN = load('..\Big Data\bla.mat')
%%
ds = datastore('C:\Users\ANTHO\Desktop\dataset_for_class\2m_y_train_set.txt','delimiter','\n')
ds2 = datastore('C:\Users\ANTHO\Desktop\dataset_for_class\2m_x_train_set.txt','delimiter','\n','ReadVariableNames',false)
%%
param.paramNbHeaders = 1;       % Minimum number of times that a word must appear to be selected in the dictionnary
param.stemming = 0;             % 1: Use stemming                  0: Doesn't use stemming
param.rmStop = 0;               % 1: Remove stops word             0: Doesn't remove it
param.bigram = 0;               % 1: Use featurize_bigram          0: Use featurize


%%
    %chunkClass = read(ds);
    %chunkData = read(ds2);
   indexSum = 1;  
while hasdata(ds)
    % Read in Chunk
    data.Class = table2array(read(ds));
    data.Data = table2cell(read(ds2));
    if length(data.Data) == 20000
        length(data.Class);
        length(data.Data);
        index = 1;
        for i = 1:length(data.Data)
           if(data.Class(i) == 1)%Negative sentence
               test.Data(index) = data.Data(i);
               test.Class(index) = -1;
               index = index + 1;
           elseif(data.Class(i) == 3)%Neutral sentence
               test.Data(index) = data.Data(i);
               test.Class(index) = 0;
               index = index + 1;
           elseif(data.Class(i) == 5)%Positive sentence
               test.Data(index) = data.Data(i);
               test.Class(index) = 1;
               index = index + 1;
           end
        end
    end

               
    
   
    wordCountVal = featurizeTest(test.Data',heads,param.rmStop,param.stemming); %Extract features from test set

    
%   %% NB
%   
%     labelVal = predict(MdlNB.Mdl,wordCountVal);
%     labelVal = str2double(labelVal)
%     accuracySummaryNB.accuracy(indexSum) = sum(labelVal'==test.Class)/length(test.Class)
%     
%     agreePos = sum(labelVal'+test.Class==2);  %Both are 1
%     valPos = sum(labelVal==1);
%     GTPos = sum(test.Class==1);
%     accuracySummaryNB.precisionPos(indexSum) = agreePos/valPos;
%     accuracySummaryNB.recallPos(indexSum) = agreePos/GTPos;
%     
%     
%     agreeNeu = sum(abs(labelVal')+abs(test.Class)==0);  %Both are 0
%     valNeu = sum(labelVal==0);
%     GTNeu = sum(test.Class==0);
%     accuracySummaryNB.precisionNeu(indexSum) = agreeNeu/valNeu;
%     accuracySummaryNB.recallNeu(indexSum) = agreeNeu/GTNeu;
% 
%     agreeNeg = sum(labelVal'+test.Class==-2);  %Both are -1
%     valNeg = sum(labelVal==-1);
%     GTNeg = sum(test.Class==-1);
%     accuracySummaryNB.precisionNeg(indexSum) = agreeNeg/valNeg;
%     accuracySummaryNB.recallNeg(indexSum) = agreeNeg/GTNeg;
%     
%     recall = [accuracySummaryNB.recallPos accuracySummaryNB.recallNeu accuracySummaryNB.recallNeg];
%     precision = [accuracySummaryNB.precisionPos accuracySummaryNB.precisionNeu accuracySummaryNB.precisionNeg];
%     accuracySummaryNB.F1(indexSum) = 2*(recall.*precision)./(recall+precision);
%     %% DT
%     labelVal = predict(MdlDT.Mdl,wordCountVal);
%     %labelVal = str2double(labelVal)
%     accuracySummaryDT.accuracy(indexSum) = sum(labelVal'==test.Class)/length(test.Class)
%     
%     agreePos = sum(labelVal'+test.Class==2);  %Both are 1
%     valPos = sum(labelVal==1);
%     GTPos = sum(test.Class==1);
%     accuracySummaryDT.precisionPos(indexSum) = agreePos/valPos;
%     accuracySummaryDT.recallPos(indexSum) = agreePos/GTPos;
% 
%     agreeNeu = sum(abs(labelVal')+abs(test.Class)==0);  %Both are 0
%     valNeu = sum(labelVal==0);
%     GTNeu = sum(test.Class==0);
%     accuracySummaryDT.precisionNeu(indexSum) = agreeNeu/valNeu;
%     accuracySummaryDT.recallNeu(indexSum) = agreeNeu/GTNeu;
%     
%     agreeNeg = sum(labelVal'+test.Class==-2);  %Both are -1
%     valNeg = sum(labelVal==-1);
%     GTNeg = sum(test.Class==-1);
%     accuracySummaryDT.precisionNeg(indexSum) = agreeNeg/valNeg;
%     accuracySummaryDT.recallNeg(indexSum) = agreeNeg/GTNeg;
%     
%     recall = [accuracySummaryDT.recallPos accuracySummaryDT.recallNeu accuracySummaryDT.recallNeg];
%     precision = [accuracySummaryDT.precisionPos accuracySummaryDT.precisionNeu accuracySummaryDT.precisionNeg];
%     accuracySummaryDT.F1(indexSum) = 2*(recall.*precision)./(recall+precision);
    
    %% NN
  
%     labelVal = predict(MdlNN.Mdl,wordCountVal);
%     labelVal = str2double(labelVal)
%     accuracySummaryNN.accuracy(indexSum) = sum(labelVal'==test.Class)/length(test.Class)
    testYclassify = MdlNN.network(wordCountVal');
    labelVal = (vec2ind(testYclassify)-2);
    accuracySummaryNN.accuracy(indexSum) = sum(labelVal==test.Class)/length(test.Class);
    
    agreePos = sum(labelVal+test.Class==2);  %Both are 1
    valPos = sum(labelVal==1);
    GTPos = sum(test.Class==1);
    accuracySummaryNN.precisionPos(indexSum) = agreePos/valPos;
    accuracySummaryNN.recallPos(indexSum) = agreePos/GTPos;

    
    agreeNeu = sum(abs(labelVal)+abs(test.Class)==0);  %Both are 0
    valNeu = sum(labelVal==0);
    GTNeu = sum(test.Class==0);
    accuracySummaryNN.precisionNeu(indexSum) = agreeNeu/valNeu;
    accuracySummaryNN.recallNeu(indexSum) = agreeNeu/GTNeu;
    
    agreeNeg = sum(labelVal+test.Class==-2);  %Both are -1
    valNeg = sum(labelVal==-1);
    GTNeg = sum(test.Class==-1);
    accuracySummaryNN.precisionNeg(indexSum) = agreeNeg/valNeg;
    accuracySummaryNN.recallNeg(indexSum) = agreeNeg/GTNeg;
    
    recall = [accuracySummaryNN.recallPos accuracySummaryNN.recallNeu accuracySummaryNN.recallNeg];
    precision = [accuracySummaryNN.precisionPos accuracySummaryNN.precisionNeu accuracySummaryNN.precisionNeg];
    accuracySummaryNN.F1Pos(indexSum) = 2*(recall(1)*precision(1))/(recall(1)+precision(1));
    accuracySummaryNN.F1Neu(indexSum) = 2*(recall(2)*precision(2))/(recall(2)+precision(2));
    accuracySummaryNN.F1Neg(indexSum) = 2*(recall(3)*precision(3))/(recall(3)+precision(3));

%     %% KNN
%     labelVal = predict(MdlKNN.Mdl,wordCountVal);
%     %labelVal = str2double(labelVal)
%     accuracyKNN(indexSum) = sum(labelVal'==test.Class)/length(test.Class)
%     
%     agreePos = sum(labelVal+test.Class==2);  %Both are 1
%     valPos = sum(labelVal==1);
%     GTPos = sum(test.Class==1);
%     precisionPosKNN(indexSum) = agreePos/valPos;
%     recallPosKNN(indexSum) = agreePos/GTPos;
%     
%         agreeNeu = sum(abs(labelVal')+abs(test.Class)==0);  %Both are 0
%     valNeu = sum(labelVal==0);
%     GTNeu = sum(test.Class==0);
%     accuracySummaryKNN.precisionNeu(indexSum) = agreeNeu/valNeu;
%     accuracySummaryKNN.recallNeu(indexSum) = agreeNeu/GTNeu;
% 
%     agreeNeg = sum(labelVal+test.Class==0);  %Both are -1
%     valNeg = sum(labelVal==0);
%     GTNeg = sum(test.Class==0);
%     precisionNegKNN(indexSum) = agreeNeg/valNeg;
%     recallNegKNN(indexSum) = agreeNeg/GTNeg;
%     
%     recall = [accuracySummaryKNN.recallPos accuracySummaryKNN.recallNeu accuracySummaryKNN.recallNeg];
%     precision = [accuracySummaryKNN.precisionPos accuracySummaryKNN.precisionNeu accuracySummaryKNN.precisionNeg];
%     accuracySummaryKNN.F1(indexSum) = 2*(recall.*precision)./(recall+precision);
    
    
%     %% NN
%     testYclassify = MdlNN.network(wordCountVal');
%     labelVal = (vec2ind(testYclassify)-2);
%     accuracyNN(indexSum) = sum(labelVal'==test.Class)/length(test.Class)
%     
%     agreePos = sum(labelVal+test.Class==2);  %Both are 1
%     valPos = sum(labelVal==1);
%     GTPos = sum(test.Class==1);
%     precisionPosNN = agreePos/valPos;
%     recallPosNN(indexSum) = agreePos/GTPos;
% 
%     agreeNeg = sum(labelVal+test.Class==0);  %Both are 0
%     valNeg = sum(labelVal==0);
%     GTNeg = sum(test.Class==0);
%     precisionNegNN = agreeNeg/valNeg;
%     recallNegNN(indexSum) = agreeNeg/GTNeg;
%     
%     F1NN(indexSum) = 2*(recall.*precision)./(recall+precision);
%     
%     
    indexSum = indexSum + 1
end
    

     save('accuracySummaryNNdddd.mat','accuracySummaryNN')
%      save('accuracySummaryNB.mat','accuracySummaryNB')
%      save('accuracySummaryDT.mat','accuracySummaryDT')
%      save('accuracySummaryKNN.mat','accuracySummaryKNN')
     