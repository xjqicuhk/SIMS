dirs = dir('../traindata/label_refine/*.png');
dirs = struct2cell(dirs);

mkdir('../traindata/label_refine_512/');
for i = 1:size(dirs,2)
    img = imread(['../traindata/label_refine/' dirs{1,i}]);
    img = imresize(img,[512,1024],'nearest');
    imwrite(img,['../traindata/label_refine_512/' dirs{1,i}]);
end