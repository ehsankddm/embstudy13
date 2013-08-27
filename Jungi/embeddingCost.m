function [cost, grad] = embeddingCost(theta, X, corruptX, embeddingSize, vocabSize, hiddenSize)

[numwords, M] = size(X);
expansion_matrix = speye(vocabSize);

% parameters learned so far
embedding = reshape(theta(1:vocabSize*embeddingSize), embeddingSize, vocabSize);
embed_to_hidden_weights = reshape(theta(vocabSize*embeddingSize+1:vocabSize*embeddingSize+numwords*embeddingSize*hiddenSize), hiddenSize, numwords*embeddingSize);
embed_to_hidden_bias = theta(vocabSize*embeddingSize+numwords*embeddingSize*hiddenSize+1:vocabSize*embeddingSize+numwords*embeddingSize*hiddenSize+hiddenSize);
score_weights = theta(vocabSize*embeddingSize+numwords*embeddingSize*hiddenSize+hiddenSize+1:end);

% gradients to be learned
embedGrad = zeros(size(embedding));
embed_to_hidden_weightsGrad = zeros(size(embed_to_hidden_weights));
embed_to_hidden_biasGrad = zeros(size(embed_to_hidden_bias));
scoreGrad = zeros(size(score_weights));

% compute embedding of the corrupted data 
corruptEmbedded =  reshape(embedding(:,corruptX(:)),  numwords * embeddingSize, M);
[corruptHiddenOutput, deriv_corruptHiddenOutput] = f(bsxfun(@plus, embed_to_hidden_weights*corruptEmbedded, embed_to_hidden_bias));
% compute embedding of the original data 
trueEmbedded = reshape(embedding(:,X(:)),  numwords * embeddingSize, M);
[trueHiddenOutput, deriv_trueHiddenOutput] = f(bsxfun(@plus, embed_to_hidden_weights*trueEmbedded, embed_to_hidden_bias));

[scoresDiff,score_func_deriv] = scoreF(score_weights, trueHiddenOutput, corruptHiddenOutput);
[errors,loss_func_deriv] = lossFunction(scoresDiff);

error_data_index = errors > 0 ;

if all(error_data_index) == 0
    cost = 0;
    grad = zeros(size(theta));
    return;
end

errors(~error_data_index) = [];
cost = 1/sum(error_data_index)*sum(errors,2);

if nargout > 1
    delta = 1/sum(error_data_index)*loss_func_deriv ;
    scoreGrad = score_func_deriv * delta' ;

    delta_true = -delta;
    delta_corrupt = delta;

    delta_true =  (score_weights * delta_true) .* deriv_trueHiddenOutput ;
    delta_corrupt = (score_weights * delta_corrupt) .* deriv_corruptHiddenOutput ;

    embed_to_hidden_weightsGrad_true = delta_true * trueEmbedded';
    embed_to_hidden_weightsGrad_corrupt = delta_corrupt * corruptEmbedded';

    embed_to_hidden_weightsGrad = embed_to_hidden_weightsGrad_true + embed_to_hidden_weightsGrad_corrupt ;
    embed_to_hidden_biasGrad = sum(delta_true + delta_corrupt,2) ;

    for n=1:numwords
%       for m=1:M
%           embedGrad(:,X(n,m)) = embedGrad(:,X(n,m)) + (embed_to_hidden_weights(:, (embeddingSize*(n-1)+1):embeddingSize*n)' * delta_true(:,m))
%           embedGrad(:,corruptX(n,m)) = embedGrad(:,corruptX(n,m)) +(embed_to_hidden_weights(:, (embeddingSize*(n-1)+1):embeddingSize*n)' * delta_corrupt(:,m))
%       end
        embedGrad = embedGrad + embed_to_hidden_weights(:, (embeddingSize*(n-1)+1):embeddingSize*n)' * delta_true * expansion_matrix(X(n,:)',:) ;
        embedGrad = embedGrad + embed_to_hidden_weights(:, (embeddingSize*(n-1)+1):embeddingSize*n)' * delta_corrupt * expansion_matrix(corruptX(n,:)',:) ;
    end
end

grad = [ embedGrad(:) ; embed_to_hidden_weightsGrad(:) ; embed_to_hidden_biasGrad(:) ; scoreGrad(:) ];

end

function [fVal, dfVal] = f(X)
    fVal = tanh(X);
    if nargout > 1
        dfVal = (1+fVal).*(1-fVal);
    end
end

function [scoreVal, dscoreVal] = scoreF(U,X,Y)
    scoreVal = 1 - U'*X + U'*Y ;
    if nargout > 1
        dscoreVal = -X + Y ;
    end
end

function [lossVal, dlossVal] = lossFunction(X)
    lossVal = max ( 0, X ) ;
    if nargout > 1
        dlossVal(lossVal ~= 0)=1;
    end
end
