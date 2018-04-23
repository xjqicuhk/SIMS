path = '../traindata/original_segment/';
for i = 1:19
    save_folder = ['../testdata/original_mask/' sprintf('%02d',i) '/'];
    mkdir(save_folder);
     
    dirs = dir(['../data/original_segment/' sprintf('%02d',i),'/*.mat']);
    dirs = struct2cell(dirs);
    for j = 2976:size(dirs,2)
        delete(['../data/original_segment/' sprintf('%02d',i),'/' dirs{1,j}])
        %load(['../data/original_segment/' sprintf('%02d',i),'/' dirs{1,j}]);
        %save([save_folder dirs{1,j}],'library_mask','library_mask_pole');
        
    end
    
end
