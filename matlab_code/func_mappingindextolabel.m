
function [label] = func_mappinglindextolabel(label_index,mapping)

 label = zeros(size(label_index,1),size(label_index,2),3,'uint8');
 label = reshape(label,[size(label_index,1)*size(label_index,2),3]);
 index = find(label_index==255);
 label(index,1) = 0;
 label(index,2) = 0;
 label(index,3) = 0;
    
 for c = 1:19
     index = find(label_index==c-1);
     label(index,1) = mapping(c,1);
     label(index,2) = mapping(c,2);
     label(index,3) = mapping(c,3);
 end
   
   label = uint8(reshape(label,[size(label_index,1),size(label_index,2),3]));
   %imwrite(label_index,colormap,[save_folder,dirs{1,i}]);
%    imshow(label_index,colormap);
%    pause;
end