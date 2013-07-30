k= 5;
load model_10_epoch.mat;
trigrams={'government of united', 'city of new', 'life in the', 'he is the', 'who is the', 'the president said'};

for trg_idx=1:size(trigrams,2)
    trigram= char(trigrams(trg_idx));
    fprintf('prediction for \" %s \"\n',trigram);
    [word1, rem]= strtok(trigram);
    [word2, rem]=strtok(rem);
    [word3, rem]=strtok(rem);
    predict_next_word(word1, word2, word3, my_model, k)
    
    fprintf('\n\n')
end



