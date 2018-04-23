load('order_frequency.mat');

sum_col = sum(frequency,1); % class 1-19 front prob
sum_row = sum(frequency,2); % class 1-19 back prob

data = cat(1,sum_col,sum_row');
data = data./repmat(sum(data,1),[2,1]);
[val,ind] = sort(data(2,:),'descend');
order = ind;
save('order.mat','order');
pause;