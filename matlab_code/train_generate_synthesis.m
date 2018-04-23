%resolution 256x512: img_path = '../data/RGB256Full/'
img_path = '../traindata/RGB512Full/';
label_path = '../traindata/label_refine/';


%list: training: 1:2975 

list = dir([img_path '*.png']);
list = struct2cell(list);

library_path_all = '../traindata/original_segment/';

% path to save all the training data
save_library_mask_path = '../traindata/synthesis/traindata_synthesis_512_1024/traindata_mat/';
%save_library_mask_img_path = '../traindata/synthesis/traindata_synthesis_512_1024/traindata_img/';
save_library_label_path = '../traindata/synthesis/traindata_synthesis_512_1024/traindata_label/';

mkdir(save_library_mask_path);
%mkdir(save_library_mask_img_path);
mkdir(save_library_label_path)

% precomputed order for cityscape dataset
order_list = load('order.mat');
order_list = order_list.order;
load('./mapping.mat');

% set img_size_h = 256(or 1024), img_size_w = 512(or 2048) if want to work on resolution 256
% x512(or 1024 x 2048) 
img_size_h = 512;
img_size_w = 1024;

% boundary dilation threshold
se = strel('square', floor(0.05*img_size_h));
% generate 5 random training sets
num_files = 5;
% color transfer augmentation
threshold = 0.8;
load('cityscapes_colormap.mat');
addpath(genpath('./colour-transfer-master/'));
% whether to visulize the result or not
visulize = 0;
%% create training folder
if(num_files>0)
    for i = 1:num_files
     save_library_mask_path_final = [save_library_mask_path sprintf('%02d',i),'/'];
    % save_library_mask_img_path_final = [save_library_mask_img_path sprintf('%02d',i),'/'];
     save_library_label_path_final = [save_library_label_path sprintf('%02d',i),'/'];
     mkdir(save_library_mask_path_final);
    % mkdir(save_library_mask_img_path_final);
     mkdir(save_library_label_path_final);
    end
end
%%

for i = 1:size(list,2)
     i  
  if(num_files > 1)
    
     img = imread([img_path, list{1,i}]);
     img = im2double(img);
     label = imread([label_path list{1,i}]);
     label = imresize(label,[img_size_h,img_size_w],'nearest');
     label_index_origin = func_mappinglabeltoindex(label, mapping);
     label_index_origin = single(label_index_origin) + 1;
     
     
    for n_f = 1:num_files
      
     save_library_mask_path_final = [save_library_mask_path sprintf('%02d',n_f), '/'];
    % save_library_mask_img_path_final = [save_library_mask_img_path sprintf('%02d',n_f), '/'];
     save_library_label_path_final = [save_library_label_path, sprintf('%02d',n_f) '/'];
 
     label_revised = label;
     label_revised = single(label_revised);
     label_revised = reshape(label_revised,[img_size_h*img_size_w,3]);
     proposal = zeros(size(img,1),size(img,2),3,'single');
     mask = zeros(size(img,1),size(img,2),'single');
    
       for c_ind = 1:19
        c = order_list(c_ind);
        library_path = [library_path_all sprintf('%02d',c) '/'];
        query = load([library_path list{1,i}(1:end-4) '.mat']);
        query_mask = query.library_mask;
        query_img = query.library_img;
        answer = zeros(size(img,1),size(img,2),3,size(query_mask,3),'double');
        query_mask = single(query_mask);
        answer_img = zeros(size(img,1),size(img,2),3,size(query_mask,3),'single');
        answer_mask = zeros(size(img,1),size(img,2),size(query_mask,3),'single');
        for j = 1:size(query_mask,3)
            query_mask_upsample = imresize(query_mask(:,:,j),[img_size_h,img_size_w],'nearest');
           
            mask = mask + query_mask_upsample;
            count  = 0;
            flag_revise = 0;
          if(sum(sum(query_mask(:,:,j)))>1000)
               summation = 0;
               current_index = 0.0;
               flag_revise = 1;
              
             %% retrive the matched segments by checking randomly search 100 times
             %  keep the segments with the largest IoU
             %  mask out the original segment with the mask provided by the
             %  searched segment
             while( count<100)
               
                    k = randperm(2975,1);
                    if(k==i) continue; end
                    response = load([library_path list{1,k}(1:end-4) '.mat']);
                    response_mask = response.library_mask;
                    response_mask = single(response_mask);
                    response_img = response.library_img;
                    current_img = im2double(imread([img_path, list{1,k}]));
                 
                    response_label_index = imread([label_path list{1,k}(1:end-4) '.png']);
                    response_label_index = imresize(response_label_index,[img_size_h,img_size_w],'nearest');
                    response_label_index = func_mappinglabeltoindex(response_label_index,mapping);
                    response_label_index = imresize(response_label_index,[img_size_h,img_size_w],'nearest');
                    response_label_index = single(response_label_index) + 1;
                   for m = 1:size(response_mask,3)
                      
                      iou = sum(sum(and(response_mask(:,:,m),query_mask(:,:,j))))/sum(sum(or(response_mask(:,:,m),query_mask(:,:,j))));
                       if(iou>summation)
                          response_mask_upsample = imresize(response_mask(:,:,m),[img_size_h,img_size_w],'nearest');
                          summation = iou;
                          current_index = m;
                          
                          valid_mask = zeros(img_size_h,img_size_w,'single');
                          valid_mask((response_mask_upsample==1)&(query_mask_upsample==1)&(response_label_index==c))=1;
                          
                          answer(:,:,:,j) = img.*repmat(valid_mask,[1,1,3]);
                          
                          answer_img(:,:,:,j)= current_img;
                       
                          answer_mask(:,:,j) = single(response_mask_upsample);


                      end
                   end
               
             count = count +1;

             end
      %% color transfer, transfer the color distribution from the searched segments
      % to the ground truth segments
             flag = rand(1);
             if(flag > threshold)
                
                 [row,col] = find(query_mask_upsample>0);
                 t1 = img(min(row(:)):max(row(:)),min(col(:)):max(col(:)),:);
                 t2 = answer_img(min(row(:)):max(row(:)),min(col(:)):max(col(:)),:,j);
                 z = colour_transfer_MKL(t1,t2);
                 z(z<0) = 0;
                 z(z>1) = 1;
               
                 answer(min(row(:)):max(row(:)),min(col(:)):max(col(:)),:,j) = z;
                 answer(min(row(:)):max(row(:)),min(col(:)):max(col(:)),:,j) =  answer(min(row(:)):max(row(:)),min(col(:)):max(col(:)),:,j).*repmat(answer_mask(min(row(:)):max(row(:)),min(col(:)):max(col(:)),j),[1,1,3]);

             end
             tmp_answer = answer(:,:,:,j);
             sum_answer = sum(single(answer(:,:,:,j)),3);
       %% boundary elision  
             sum_answer(sum_answer>0) = 1;
             boundary_answer = edge(sum_answer,'Sobel',0);
             boundary_answer = imdilate(single(boundary_answer), se);
            
            revise_index = find(boundary_answer==1);
            d = randperm(size(revise_index,1));
            revise_index = revise_index(d(1:floor(size(d,2)*0.5)));
            tmp_answer = reshape(tmp_answer,[img_size_h*img_size_w, 3]);
            revise_value = zeros(size(revise_index,1),3)+1;  
            tmp_answer(revise_index,:) = revise_value;
            tmp_answer = reshape(tmp_answer,[img_size_h, img_size_w,3]);
             
             z = single(tmp_answer);
             z= sum(z,3);
             z(z>0) = 1;
             proposal = proposal.*repmat(1-z,[1,1,3]) + tmp_answer;
            
          end
          %% label map add noise(optional)
              index1 = find(query_mask_upsample==1&label_index_origin == c);
              label_revised(index1,1) = single(mapping(c,1));
              label_revised(index1,2) = single(mapping(c,2));
              label_revised(index1,3) = single(mapping(c,3));
            
           if(flag_revise==1)
               query_mask_single = query_mask_upsample;
               %random rotate and translate
               %rotation and translation and resize
               rotate_degree = (rand(1)-0.5)*5;
               %random translation
               translate_x = (rand(1)-0.5)*size(img_size_h,1)*0.0675;
               translate_y = (rand(1)-0.5)*size(img_size_w,2)*0.0675;
               query_mask_single = imrotate(query_mask_single, rotate_degree, 'nearest');
               query_mask_single = imtranslate(query_mask_single,[translate_x,translate_y],'FillValues',0);
               
               length_r = size(query_mask_single,1);
               length_c = size(query_mask_single,2);
    
               residual_r = length_r - size(img,1) + 1;                 
               residual_c = length_c - size(img,2) + 1;
               start_r = unidrnd(residual_r,1,1);
               start_c = unidrnd(residual_c,1,1);
               end_r = start_r+size(img,1)-1;
               end_c = start_c+size(img,2)-1;
               
               query_mask_single = query_mask_single(start_r:end_r,start_c:end_c,:);
               
               index1= find(query_mask_upsample==0&query_mask_single~=0);
               label_revised(index1,1) = 0;
               label_revised(index1,2) = 0;
               label_revised(index1,3) = 0;
               
               index1= find((query_mask_upsample==0&response_mask_upsample~=0));
               label_revised(index1,1) = 0;
               label_revised(index1,2) = 0;
               label_revised(index1,3) = 0;
               
           end
           
        end
         
   end
        label_revised = reshape(label_revised,[img_size_h,img_size_w,3]);
     %% visulize the generated data
     if(visulize)
         figure(1);
         imshow(proposal);
         figure(2);
         imshow(mask);
         figure(3);
         imshow(img);
         figure(4);
         imshow(uint8(label_revised));
         figure(5);
         imshow(label);
         pause;
     end
    %% save the generated data
    save_warper([save_library_mask_path_final,list{1,i}(1:end-4) '.mat'],uint8(proposal*255.0),uint8(mask));
%    imwrite(uint8(proposal*255.0),[save_library_mask_img_path_final, list{1,i}(1:end-4) '.png']);
    imwrite(uint8(label_revised),[save_library_label_path_final list{1,i}(1:end-4) '.png']);
  
    end
  end
end