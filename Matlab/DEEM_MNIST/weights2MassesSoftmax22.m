function [F1,C,betp,pl] = weights2MassesSoftmax22(F1,optionsEVNN,applysoftmax)
% F1 are output by NN
% Use method of Thierry to compute BBAs
% apply a softmax
% Coding
% Conflict
% Singletons 
% Universe if any
% Pairs if universe and if any

% besoin de cette option pour un cas particulier, true par defaut
if nargin==2, applysoftmax=true; end

betp = []; pl = [];
numclasses = optionsEVNN.numClasses;
usedoute = optionsEVNN.usedoute;
assert(usedoute); % on peut utiliser le doute et le conflit au minimum
usepaires = optionsEVNN.usepaires;
% numData = size(F1,2);

% assert(usedoute==usepaires);
positionDoute = numclasses + 2; % conflict first
positionClasse1 = 2;
positionClasseK = positionClasse1 + numclasses - 1;
nbpairesclasses = (numclasses*(numclasses-1)/2); 
% Focal sets
fs = numclasses + 2*double(usedoute==true) + double(usepaires==true)*nbpairesclasses;
assert(fs==size(F1,1));

% Normalise outputs
if applysoftmax
    F1 = softmax(F1);
end

% Build C matrix
[C,forDecision] = buildCmatrix22(optionsEVNN);

if nargout >= 3 
    if usepaires
        
        betp = F1(positionClasse1:positionClasseK,:); % conflict first, betp init mass on singletons
        pl   = F1(positionClasse1:positionClasseK,:); % conflict first, pl init mass on singletons
        betp = betp + F1(positionDoute,:)/numclasses; % ajoute doute
        pl   = pl   + F1(positionDoute,:); % ajoute doute
        for i=1:numclasses
            d = forDecision(:,i);
            betp(i,:) = betp(i,:) + sum(d .* F1(positionDoute+1:end,:),1) / 2;
            pl(i,:)   = pl(i,:)   + sum(d .* F1(positionDoute+1:end,:),1);
        end
        n = sum(betp,1);
        betp = betp ./ n; % pignistic transform
        n = sum(pl,1);
        pl = pl ./ n; % contour function
        
    else
        
        % BETP
        betp = F1(positionClasse1:positionClasseK,:);
        if usedoute
            douteX1 = F1(positionDoute,:);
            betp = F1(positionClasse1:positionClasseK,:) + douteX1 / numclasses; % distribute universe among singletons
        end
        n = sum(betp,1);
        betp = betp ./ n; % class predicted for Xi
        
        % PL
        pl = F1(positionClasse1:positionClasseK,:);
        if usedoute
            pl = pl + douteX1;
        end
        n = sum(pl,1);
        pl = pl ./ n; % class predicted for Xi
    end
end

betp(isnan(betp)) = 1/numclasses;
pl(isnan(pl)) = 1/numclasses;

end