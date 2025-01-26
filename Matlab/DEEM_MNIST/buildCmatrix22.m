function [C,forDecision] = buildCmatrix22(optionsEVNN)
% F1new is just there to take its type
% Its content is not used

numclasses = optionsEVNN.numClasses;
usedoute = optionsEVNN.usedoute;
usepaires = optionsEVNN.usepaires;

positionClasse1 = 1;
positionClasseK = numclasses;
if usedoute %or usepaires
    positionDoute = numclasses + 2; % conflict first
    positionConflict = 1;
    positionClasse1 = 2;
    positionClasseK = numclasses + 1;
end

if usepaires

    nbpaires = (numclasses*(numclasses-1)/2);
    pairCoding = optionsEVNN.pairCoding;

    % C matrix for conflict
    % exemple with four classes
    %       C    w1   w2   w3  w4 Omega  w1w2  w1w3  w1w4  w2w3  w2w4  w3w4
    % C     1    1    1    1   1    1      1     1    1     1      1     1
    % w1    1    0    1    1   1    0      0     0    0     1      1     1
    % w2    1    1    0    1   1    0      0     1    1     0      0     1
    % w3    1    1    1    0   1    0      1     0    1     0      1     0
    % w4    1    1    1    1   0    0      1     1    0     1      0     0
    % Omega 1    0    0    0   0    0      0     0    0     0      0     0
    % w1w2  1    0    0    1   1    0      0     0    0     0      0     1
    % w1w3  1    0    1    0   1    0      0     0    0     0      1     0
    % w1w4  1    0    1    1   0    0      0     0    0     1      0     0
    % w2w3  1    1    0    0   1    0      0     0    1     0      0     0
    % w2w4  1    1    0    1   0    0      0     1    0     0      0     0
    % w3w4  1    1    1    0   0    0      1     0    0     0      0     0
    % exemple with three classes
    %       C    w1   w2   w3  Omega  w1w2  w1w3 w2w3
    %  C    1    1    1    1   1      1     1    1     
    % w1    1    0    1    1   0      0     0    1
    % w2    1    1    0    1   0      0     1    0
    % w3    1    1    1    0   0      1     0    0
    % Omega 1    0    0    0   0      0     0    0
    % w1w2  1    0    0    1   0      0     0    0
    % w1w3  1    0    1    0   0      0     0    0
    % w2w3  1    1    0    0   0      0     0    0
    
    C = ones(optionsEVNN.fs);%,'like',F1new);
    C = C .* (1-eye(optionsEVNN.fs));  % no conflict on diagonal A1 inter A2
    C(positionDoute,:) = 0;% for universe intersect diff from 0
    C(:,positionDoute) = 0;
    C(positionConflict,:) = 1;% for conflict 
    C(:,positionConflict) = 1;
    % intersection paires singletons
    for i=1:numclasses
        C(positionDoute+1:end,i+1) = not(sum(ismember(pairCoding,i),2)>0);
        C(i+1,positionDoute+1:end) = C(positionDoute+1:end,i+1)';
    end
    % intersection paires paires
    for i=1:nbpaires
        C(positionDoute+1:end,positionDoute+i) = not(sum(ismember(pairCoding,pairCoding(i,:)),2)>0);
        C(positionDoute+i,positionDoute+1:end) = C(positionDoute+1:end,positionDoute+i)';
    end
    
    % construit la BETP et CONTOUR PL pour decision ou crossentropy
    forDecision = not(C(positionDoute+1:end,positionClasse1:positionClasseK));
    forDecision = logical(forDecision);
   
    
else
    
    % C matrix for conflict
    % exemple with three classes
    %       C   w1   w2   w3  Omega
    % C     1    1    1    1    1
    % w1    1    0    1    1    0
    % w2    1    1    0    1    0
    % w3    1    1    1    0    0
    % Omega 1    0    0    0    0    
    C = ones(numclasses + 2*double(usedoute==true));%,'like',F1);
    C = C .* (1-eye(numclasses + 2*double(usedoute==true))); % no conflict on diagonal A1 inter A2
    if usedoute
        C(positionDoute,:) = 0;% for universe intersect diff from 0 at last line and
        C(:,positionDoute) = 0;% last column of C since m(Omega inter wk) ~= emptyset
        C(positionConflict,:) = 1;% conflict
        C(:,positionConflict) = 1;% conflict
    end
    forDecision = [];
    
end

end