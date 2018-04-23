
function [label_index] = func_mappinglabeltoindex(label,mapping)

 label_index = zeros(size(label,1),size(label,2),'single');
 mask_matrix = label-label;
 index_invalid = sum(single(label),3);
label_index(index_invalid==0) = 255;
 
 for c = 1:19
     mask_matrix(:,:,1) = mapping(c,1);
     mask_matrix(:,:,2) = mapping(c,2);
     mask_matrix(:,:,3) = mapping(c,3);
     corresponding_label = single(mask_matrix)-single(label);
     corresponding_label = sum(abs(corresponding_label),3);
     label_index(corresponding_label==0)= c-1;
 end
   label_index = uint8(label_index);
   
   %imwrite(label_index,colormap,[save_folder,dirs{1,i}]);
%    imshow(label_index,colormap);
%    pause;
end