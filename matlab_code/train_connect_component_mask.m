% The code is to generate the memory base.
% Input: RGB images (img_path)
% Input: Label map (label_path)
% Output: segment, corresponding mask
%% output structure:
% library_img (training data): segmented rgb patches based on connectivity
% 
% library_mask: connected segment mask, 1 indicates exist
% library_mask_pole: (optional)Special handling for cityscapes dataset,
% because many objects often separated by thin pole structures (pole,
% light, traffic sign),
% this is not very important just for sightly better results
% for other datasets, can be deleted.


%% Function: Get the corresponding segment based on connectivity analysis.

img_path = '../traindata/RGB256Full/';
label_path = '../traindata/label_refine/';
list = dir([img_path '*.png']);
list = struct2cell(list);
load('cityscapes_colormap.mat');
pixel_matrix = zeros(256,512,3,'uint8');


save_path_library = '../traindata/original_segment/';

mkdir(save_path_library);



img_size_h = 256;
img_size_w = 512;
mapping = load('mapping.mat');
mapping = mapping.mapping;
pole_matrix = zeros(img_size_h,img_size_w,3,'uint8');

%% think structure
% cityscape thin structure effect remove, we find in cityscapes, only
% single buding often split out by the pole, tranffic sign, and traffic
% light, segments with the same class that are separated by these thin
% structures are merged together
pole_matrix(:,:,1) = mapping(6,1);
pole_matrix(:,:,2) = mapping(6,2);
pole_matrix(:,:,3) = mapping(6,3);

sign_matrix = pole_matrix;
sign_matrix(:,:,1) = mapping(7,1);
sign_matrix(:,:,2) = mapping(7,2);
sign_matrix(:,:,3) = mapping(7,3);

light_matrix = pole_matrix;
light_matrix(:,:,1) = mapping(8,1);
light_matrix(:,:,2) = mapping(8,2);
light_matrix(:,:,3) = mapping(8,3);

for c = 1:19
   
    pixel_matrix = zeros(img_size_h,img_size_w,3,'uint8');
    pixel_matrix(:,:,1) = mapping(c,1);
    pixel_matrix(:,:,2) = mapping(c,2);
    pixel_matrix(:,:,3) = mapping(c,3);
    save_path_library_class = [save_path_library sprintf('%02d',c)];
    mkdir([save_path_library_class '/']);
   
    
    for i = size(list,2):size(list,2)
        i
        img = imread([img_path, list{1,i}]);
        img = im2double(img);
        label = imread([label_path list{1,i}]);
        mask = single(label)-single(pixel_matrix);
        mask = sum(mask.^2,3);
        mask(mask~=0) = -1;
        mask = mask + 1;
        mask_original = mask;
        %% sanity check for think structures, comment this if you do not use this
        % comment begin
        if(c~=6 & c~=7 & c~=8)
            mask_pole = sum((single(label)-single(pole_matrix)).^2,3);
            mask_pole(mask_pole~=0) = -1;
            mask_pole = mask_pole + 1;
            
            mask_light = sum((single(label)-single(light_matrix)).^2,3);
            mask_light(mask_light~=0) = -1;
            mask_light = mask_light + 1;
            
            mask_sign = sum((single(label)-single(sign_matrix)).^2,3);
            mask_sign(mask_sign~=0) = -1;
            mask_sign = mask_sign + 1;
            small_mask = mask_pole + mask_sign + mask_light;
            mask(mask_pole==1 | mask_light==1 | mask_sign == 1) = 1;
        end
        % comment end
        

        %% find connected component
        connected_component = bwlabel(mask);
        conn = unique(connected_component);
        
        library_img = zeros(size(img,1),size(img,2),3,size(conn,1),'uint8');
        library_mask = zeros(size(img,1),size(img,2),size(conn,1),'uint8');
        library_mask_pole = zeros(size(img,1),size(img,2),size(conn,1),'uint8');

       
        empty_ind = [];
        for j = 1:size(conn,1)
            [r, col] = find(connected_component==conn(j));
          if(c~=6 & c~=7 & c~=8) 
            %  t = size(r,1) < 1000
            if((sum(sum(small_mask(sub2ind([img_size_h,img_size_w],r,col))))>=sum(sum(mask_original(sub2ind([img_size_h,img_size_w],r,col))))))
                empty_ind(end+1) = j;
                continue;
            end
         
          end
            if(mask(r(1),col(1))==1)
               
                tmp_mask = mask;
                tmp_mask((connected_component~=conn(j))) = 0;
               
                tmp_img = img.*repmat(tmp_mask,[1,1,3]);
                tmp_img = tmp_img.*repmat(mask_original,[1,1,3]);
                library_img(:,:,:,j) = uint8(tmp_img*255.0);
                library_mask(:,:,j) = uint8(tmp_mask.*mask_original);
                [row_lm, col_lm] = find(library_mask(:,:,j)==1);
                range_row = min(row_lm(:)):max(row_lm(:));
                range_col = min(col_lm(:)):max(col_lm(:));
                library_mask_pole(range_row,range_col,j) = uint8(tmp_mask(range_row,range_col));

            end
        end
        
        library_img(:,:,:,empty_ind) = [];
        
        library_mask(:,:,empty_ind) = [];
        library_mask_pole(:,:,empty_ind) = [];
        save([save_path_library_class,'/',list{1,i}(1:end-4),'.mat'],'library_img','library_mask','library_mask_pole');
      
       
    end
end