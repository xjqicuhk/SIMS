path = '../traindata/transform/transform_512/02/';
dirs = dir([path '*.mat']);
dirs = struct2cell(dirs);

for i = 1:size(dirs,2)
    if(str2num(dirs{1,i}(1:end-4))>3000)
        delete([path dirs{1,i}]);
    end
end

