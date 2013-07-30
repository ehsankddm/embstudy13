k= 10;
load model_10_epoch.mat;
words={'companies', 'president', 'day', 'could', 'he', 'she', 'federal'}

for wrd_idx =1:size(words,2)
    fprintf('similar words to %s \n', char(words(wrd_idx)));
    display_nearest_words(words(wrd_idx), my_model, k);
    fprintf('\n\n')
end
