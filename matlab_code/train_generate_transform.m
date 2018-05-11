addpath('func_save')
img_path = '../traindata/RGB512Full/';
label_path = '../traindata/label_refine/';
list = dir([img_path '*.png']);
list = struct2cell(list);
load('cityscapes_colormap.mat');

library_path = '../traindata/original_segment/';

save_library_mask_path = '../traindata/transform/transform_512/';
mkdir(save_library_mask_path);


max_num = 30;
img_size_h = 512;
img_size_w = 1024;
num_files = 2;
num_class = 19;
num_thresh = 1000;

for j = 1:num_files
    mkdir([save_library_mask_path sprintf('%02d',j) '/']);
end

max_t = 0;
visulize = 0;
% transformation ratio can be changed according to applications.
rotate_ratio = 5;
for i =1:size(list,2)
    i
   for n = 1:num_files
    
    %tic
    img = imread([img_path, list{1,i}]);
    img = im2double(img);
    label = im2double(imread([label_path list{1,i}]));
    proposal = zeros(max_num,size(img,1),size(img,2),3,'uint8');
    proposal_gt = zeros(max_num, size(img,1),size(img,2),3,'uint8');
    mask = zeros(max_num,img_size_h,img_size_w,'uint8');
    t = 0;
    for c = 1:19
     if(~exist([library_path sprintf('%02d/',c) list{1,i}(1:end-4) '.mat'],'file')) 
         continue;
     end
    query = load([library_path sprintf('%02d/') sprintf('%02d/',c) list{1,i}(1:end-4) '.mat']);
    query_mask = query.library_mask;
    query_img = query.library_img;
   
   
    
    for j = 1:size(query_mask,3)
        count  = 0;
        
      if(sum(sum(query_mask(:,:,j)))>num_thresh)
           t = t +1;
           mask_resize = imresize(single(query_mask(:,:,j)),[img_size_h,img_size_w],'nearest');
           
           
           tmp_proposal = img.*(repmat(mask_resize,[1,1,3]));
           tmp_proposal_origin = tmp_proposal;
          
           %rotation and translation and resize
           rotate_degree = (rand(1)-0.5)*rotate_ratio;
           [row,col] = find(mask_resize ==1);
           rowmin = min(row(:)); colmin = min(col(:));
           rowmax = max(row(:)); colmax = max(col(:));
           center_row = ceil((rowmax-rowmin)/2)+rowmin; center_col = ceil((colmax-colmin)/2)+colmin;
           crop_proposal = tmp_proposal(rowmin:rowmax,colmin:colmax,:);
           crop_proposal_original = crop_proposal;
           
           original_coordinate = [rowmin,rowmax,colmin,colmax];
            
           
           
           %random scale
           %rand_scale = 1.0;
           rand_scale = (rand(1)-0.5)/5 +1.0;
           crop_proposal = imresize(crop_proposal,rand_scale,'bicubic');
           
           length_r = size(crop_proposal,1); length_c = size(crop_proposal,2);
           start_r = max(center_row - ceil(length_r/2),1); start_c = max(center_col-ceil(length_r/2),1);
           r_end = min(start_r + length_r - 1,img_size_h); c_end = min(start_c+length_c-1,img_size_w);
           
           
           if(start_r+length_r-1<=img_size_h &&start_c+length_c-1<=img_size_w && center_row - ceil(length_r/2)>=1 &&center_col-ceil(length_r/2)>=1)
           %result in non-uniform scaling in x-direction and y-direction
             tmp_proposal = tmp_proposal - tmp_proposal;
            %tmp_proposal(start_r:r_end,start_c:c_end,:)= imresize(crop_proposal,[r_end-start_r+1, c_end-start_c+1],'bilinear');
             tmp_proposal(start_r:r_end,start_c:c_end,:)= crop_proposal;
             original_coordinate = [start_r,r_end,start_c,c_end];
           end
           tmp_proposal_tmp = tmp_proposal;

            %random translation translate_x, col ++
            % translate_y, row++
           translate_x = round((rand(1)-0.5)*size(crop_proposal,2)*0.125/2);
           translate_y = round((rand(1)-0.5)*size(crop_proposal,1)*0.125/2);
           tmp_proposal = imtranslate(tmp_proposal,[(translate_x),(translate_y)],'FillValues',0);
          
           length_r = size(tmp_proposal,1);
           length_c = size(tmp_proposal,2);

           residual_r = length_r - size(img,1) + 1;                 
           residual_c = length_c - size(img,2) + 1;
           start_r = unidrnd(residual_r,1,1);
           start_c = unidrnd(residual_c,1,1);
           end_r = start_r+size(img,1)-1;
           end_c = start_c+size(img,2)-1;
           
           if(original_coordinate(1)+translate_y>=1&&original_coordinate(2)+translate_y<=img_size_h...
               && original_coordinate(3)+translate_x>=1&&original_coordinate(4)+translate_x<=img_size_w)
           tmp_proposal = tmp_proposal(start_r:end_r,start_c:end_c,:);
           else
           tmp_proposal = tmp_proposal_tmp;
           end
           tmp_proposal = uint8(tmp_proposal*255.0);
         % proposal .* original mask
           proposal(t,:,:,:) = tmp_proposal;
           mask(t,:,:) = uint8(mask_resize);
           proposal_gt(t,:,:,:) = uint8(tmp_proposal_origin*255.0); 
           if(visulize)
               figure(1);
               imshow(squeeze(proposal(t,:,:,:)));
               figure(2);
               imshow(squeeze(proposal_gt(t,:,:,:)));
               figure(3);
               imshow(squeeze(mask(t,:,:)),[]);
               pause;
           end
        
      end
    end
    end
%     if(t>max_t)
%         max_t = t;
%     end
   proposal = proposal(1:t,:,:,:);
   mask = mask(1:t,:,:,:);
   proposal_gt = proposal_gt(1:t,:,:,:);
   
   save_warper_2([save_library_mask_path, sprintf('%02d/',n) list{1,i}(1:end-4) '.mat'],proposal,mask,proposal_gt);
   
   
end
end