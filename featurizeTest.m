function [ featureVector ] = featurizeTest( inputcellarray, headers, removeStopWords, doStem )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

outputMatrix = zeros(size(inputcellarray,1),length(headers));
for i = 1:size(inputcellarray,1)
    %fprintf('%d/%d ', i, size(inputcellarray,1));
    comment = inputcellarray{i};
    comment = SanitizeComment(comment);
    comment = lower(comment);
    
    r=regexp(comment,' ','split');
    comment = [];
    for j =1:size(r,2)
        if doStem
            word = porterStemmer(cell2mat(r(j)));
        else
            word = (cell2mat(r(j)));
        end
        
        comment = [comment,' ',word];
    end
    outputMatrix(i,:) = term_count(comment, headers);
    
    if mod(i,300)==0
        a = sprintf('%d', i);
        disp(a)
    end
    
    
    
end
featureVector=outputMatrix;
end

