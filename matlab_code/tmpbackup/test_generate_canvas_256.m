addpath(genpath('./func_save'));
addpath(genpath('./colour-transfer-master'));
% Refined coarse label path
label_coarse_path = '../data/Label256Full_pred_refine/';
% Original training image label path (fine data)
label_path = '../data/Label256Full/';

% Full list of the whole cityscape dataset
% Train: 1-2975
% Test: 00100001-00100500 (2976-3475)
list = dir([label_path '*.png']);
list = struct2cell(list);

% Order data path
% Order original data
path_proposal_order = '../testdata/testdata_order_all/';
% Order data network output
path_proposal_score = '../python_code/model/order/data/';

% Retrieved segments before transformation
path_response_proposal = '../testdata/transform/transform_256/';
% Segments after transformation
path_transferred = '../python_code/model/spatial_transformer_cityscape_512/transform_data_iou_max_all_512_order_transformed/';

save_response_path = '../testdata/synthesis/transform_order_512/';
save_response_img = '../testdata/synthesis/transform_order_512_canvas/';
%save_revised_label = '../testdata/transferred_512_order_coarse_l_order/transferred_label/';

mkdir(save_response_path);
mkdir(save_response_img);
%mkdir(save_revised_label);

img_size_h = 512;
img_size_w = 1024;
se = strel('square', floor(0.025*img_size_h));
load('./mapping.mat');
% 11: sky, 1: road, 2:sidewalk, 3:building, 4:wall, 5:fence, 9: vegetation, 10: terrian, 6: pole, 7:traffic light
% 8: traffic sign, 14: car, 15: truck, 16:bus, 17: train, 18: motorcycle,
% 19: bicycle, 12: person, 13: rider
%%

context_class = [1,2,4,5,9,10,11];
object_class = [3,6,7,8,12,13,14,15,16,17,18,19];
% deshadow operation
modify_boundary = 1;
se_road = strel('square', 31);
modify_tree = 1;
fill_context = 1;
se_tree = strel('square',21);
%test phase set to 0.5
elision_ratio = 0.5;
invalid_class = 256;
se_outer = strel('square', 41);
extension_width = 0.125;

visulize = 0;


%%
parfor i = 2976:size(list,2)

    i
  %  img = imread([img_path, list{1,i}]);
   % img = im2double(img);
    label = imread([label_coarse_path list{1,i}]);
   
    label_index_origin = func_mappinglabeltoindex(label,mapping);
    label_index_origin = single(label_index_origin) + 1;
    
    label = imresize(label,[img_size_h,img_size_w],'nearest');
    label_index_origin = imresize(label_index_origin,[img_size_h,img_size_w], 'nearest');
    
    data = load([path_response_proposal list{1,i}(1:end-4) '.mat']);
    answer = data.proposal;
    answer_iou = data.proposal_iou;
    answer_pole_mask = data.proposal_pole_mask;
    response_index = data.response_index;
    original_index = data.original_index;
    
    mask_remained = data.mask_remained;
    class_remained = data.class_remained;
    original_index_remained = data.original_index_remained;
    
    query_mask = single(data.mask);
    mask_remained = single(mask_remained);
    all_c = data.class;
    all_class_flag = data.all_class_flag;
    
    transferred_proposal = load([path_transferred list{1,i}(1:end-4) '.mat']);
    transferred_proposal = transferred_proposal.transferred;
    % order data
    proposal_order_mask_data = load([path_proposal_order list{1,i}(1:end-4) '.mat']);
    proposal_order_mask = proposal_order_mask_data.semantic_segment_mask; %segment
    proposal_order_index = proposal_order_mask_data.semantic_segment_label;%[j,k,c1,c2]c1: number j, c2:number k
    proposal_order_pred = load([path_proposal_score list{1,i}(1:end-4) '.mat']);
    proposal_order_pred = proposal_order_pred.prediction;
    % transferred_proposal = answer;
    
    %transferred_proposal = answer;
    
    
    
    proposal = zeros(img_size_h,img_size_w,3,'single');
    new_label= zeros(img_size_h,img_size_w,3,'uint8');
    fill_region = zeros(img_size_h,img_size_w,2,'single');
    total_num = size(all_class_flag,1);
    for l = 1:total_num
        
        
       if(all_class_flag(l,1)==1)
        j = all_class_flag(l,2);
        c = all_c(j);
        proposal_pole_mask = single(squeeze(answer_pole_mask(j,:,:)));
        
        
       %% transformation patch: check the patch and refine the label map according to the selected proposal
        
          label_previous =label;
          
          label = single(label);
          label = reshape(label,[img_size_h*img_size_w,3]);
        
          index1 = find(squeeze(query_mask(j,:,:))==1|proposal_pole_mask==1);
          label(index1,1) = 0;
          label(index1,2) = 0;
          label(index1,3) = 0;
          
          %check failure mode of spatial transformer if make iou decrease
          %delete this transformation
          t = single(transferred_proposal(j,:,:,:));
          t = squeeze(t);
          t = sum(t,3);
          t(t>0) = 1;
          
          t_origin = single(answer(j,:,:,:));
          t_origin = squeeze(t_origin);
          t_origin = sum(t_origin,3);
          t_origin(t_origin>0) = 1;
          
          if(sum(sum(t&squeeze(query_mask(j,:,:))))<=sum(sum(t_origin&squeeze(query_mask(j,:,:)))))
              transferred_proposal(j,:,:,:) = answer(j,:,:,:);
              t = t_origin;
          end
          %% check order
          %% Check order
         if(~ismember(c,[-1]))
             index1 = find(query_mask(:,:,j)==1|answer_pole_mask==1);
             % index = find(t==1);
             filled_overlap = single(fill_region).*repmat((t==1),[1,1,2]);
             filled_overlap = reshape(filled_overlap,[img_size_h*img_size_w,2]);
             overlap_mask = sum(filled_overlap,2);
             overlap_mask(overlap_mask>0) = 1;
             unique_pair = unique(filled_overlap,'rows');
             invalid = sum(unique_pair,2);
             unique_pair(invalid==0,:) = [];%(j,k,c1,c2)
             current_mask = zeros(img_size_h,img_size_w,'single');
             overlap_mask = reshape(overlap_mask,[img_size_h,img_size_w]);
             filled_overlap = reshape(filled_overlap,[img_size_h,img_size_w,2]);
             for o = 1:size(unique_pair,1)
              set = double([unique_pair(o,1),j,unique_pair(o,2),c]);
              diff = proposal_order_index - repmat(set,[size(proposal_order_index,1),1]);
              diff = sum(abs(diff),2);
              diff_index = find(diff==0);
              
               if(~isempty(diff_index))
                  prediction_order = proposal_order_pred(diff_index,:);
                  [val,ind] = max(prediction_order);
                  ind = ind-1;
                  if(ind == unique_pair(o,2))
                 % current_set = set(diff_index,:);
                  current_mask_origin = (filled_overlap - repmat(reshape(unique_pair(o,:),[1,1,2]),[img_size_h,img_size_w,1])).*repmat(overlap_mask,[1,1,2]);
                  current_mask_origin = sum(abs(current_mask_origin),3);
                  current_mask_origin(current_mask_origin>0)=1;
                  current_mask = current_mask + current_mask_origin;
                  end
               end
             end
             %current_mask_resize = imresize(current_mask,[img_size_h,img_size_w],'nearest');
            
             t = t.*(1-current_mask);
             tmp = single(squeeze(transferred_proposal(j,:,:,:)));
             tmp = tmp.*repmat(1-current_mask,[1,1,3]);
             transferred_proposal(j,:,:,:) = uint8(tmp);
         end
         %% change label
          index = find(t==1);
          label(index,1) = single(mapping(c,1));
          label(index,2) = single(mapping(c,2));
          label(index,3) = single(mapping(c,3));
          [label_row,label_col] = find(t==1);
          label = reshape(label,[img_size_h,img_size_w,3]);
          fill_region = reshape(fill_region,[img_size_h*img_size_w,2]);
          fill_region(index,1) = original_index(j);
          fill_region(index,2) = c;
          fill_region = reshape(fill_region,[img_size_h,img_size_w,2]);
        
        %% processing the proposal when shadow exist in the proposal,
        % shadow operation vehicle and road, de-original shadow
        if(modify_boundary&(c==1|c==2))
            tmp_answer_img = squeeze(transferred_proposal(j,:,:,:));
            label_index_response = imread([label_path,list{1,response_index(j)}(1:end-4) '.png']);
            label_index_response = func_mappinglabeltoindex(label_index_response,mapping);
            label_index_response = single(label_index_response) + 1;
            %label_index_response = imresize(label_index)
            mask_modify_boundary = zeros(img_size_h,img_size_w,'single');
            mask_modify_boundary(label_index_response==14|label_index_response==15|label_index_response==16|label_index_response==17) =1;
            modify_boundary_mask = edge(mask_modify_boundary,'Sobel',0);
            modify_boundary_mask = imdilate(single(modify_boundary_mask), se_road);
            index = find(modify_boundary_mask==1&(modify_boundary_mask==1 | modify_boundary_mask==2)&t==1);
            tmp_answer_img = reshape(tmp_answer_img,[img_size_h*img_size_w,3]);
            tmp_answer_img(index,1) = 0;
            tmp_answer_img(index,2) = 0;
            tmp_answer_img(index,3) = 0;
            transferred_proposal(j,:,:,:) = reshape(tmp_answer_img,[img_size_h,img_size_w,3]);    
        end
        % tree patches inaccurate label include large amount of sky
        % pixels.
        %% (optional trick with tree regions): tree segment often include sky, so larger dilation
        if(modify_tree&c==9)
            tmp_answer_img = squeeze(transferred_proposal(j,:,:,:));
            mask_modify_boundary = sum(single(tmp_answer_img),3);
            mask_modify_boundary(mask_modify_boundary>0) = 1;
            modify_boundary_mask = edge(mask_modify_boundary,'Sobel',0);
            modify_boundary_mask = imdilate(single(modify_boundary_mask), se_tree);
            index = find(modify_boundary_mask==1);
            tmp_answer_img = reshape(tmp_answer_img,[img_size_h*img_size_w,3]);
            tmp_answer_img(index,1) = 0;
            tmp_answer_img(index,2) = 0;
            tmp_answer_img(index,3) = 0;
            transferred_proposal(j,:,:,:) = reshape(tmp_answer_img,[img_size_h,img_size_w,3]);
        end
        
        
        %% inner elision regions
         tmp_answer = im2double(transferred_proposal(j,:,:,:));
         sum_answer = sum(squeeze(single(transferred_proposal(j,:,:,:))),3);
         sum_answer(sum_answer>0) = 1;
         boundary_answer = edge(sum_answer,'Sobel',0);
         boundary_answer = imdilate(single(boundary_answer), se);
         revise_index = find(boundary_answer==1&sum_answer>0);
         d = randperm(size(revise_index,1));
         revise_index = revise_index(d(1:floor(size(d,2)* elision_ratio)));
         tmp_answer = reshape(tmp_answer,[img_size_h*img_size_w, 3]);
         revise_value = zeros(size(revise_index,1),3)+1;  
         tmp_answer(revise_index,:) = revise_value;
         tmp_answer = reshape(tmp_answer,[img_size_h, img_size_w,3]);
         z = single(tmp_answer);
         z= sum(z,3);
         z(z>0) = 1; 
        %% paste the proposal
         z = z+t;
         z(z>0) = 1;
             
         proposal = proposal.*repmat(1-z,[1,1,3]) + tmp_answer;
        
   
        
         elseif(all_class_flag(l,1)==2)
          %% label unchanged id no propper proposal
          j = all_class_flag(l,2);
          label = single(label);
          c = class_remained(j);
          label = reshape(label,[img_size_h*img_size_w,3]);
          
          index1 = find(squeeze(mask_remained(j,:,:))==1);
          label(index1,1) = single(mapping(c,1));
          label(index1,2) = single(mapping(c,2));
          label(index1,3) = single(mapping(c,3));
          label = reshape(label,[img_size_h,img_size_w,3]);
          
          proposal = reshape(proposal,[img_size_h*img_size_w,3]);
          proposal(index1,1) = 0;
          proposal(index1,2) = 0;
          proposal(index1,3) = 0;
          proposal = reshape(proposal,[img_size_h,img_size_w,3]);
       end
    end
    
    
    %% check unfilled labels
    if(fill_context)
        label_mask = sum(single(label),3);
        label = reshape(label,[img_size_h*img_size_w, 3]);
        label_previous = reshape(label_previous,[img_size_h*img_size_w,3]);
        label_mask(label_mask>0) = 1;

         for ind = 1:size(context_class,2)
            index1 = find((label_index_origin == context_class(ind))&label_mask==0);
            label(index1,1) = mapping(context_class(ind),1);
            label(index1,2) = mapping(context_class(ind),2);
            label(index1,3) = mapping(context_class(ind),3); 
         end
    end
    
     %% Outer elision to make the network generate shadows object to context, downside  
     label = reshape(label,[img_size_h, img_size_w,3]);
     label_index = func_mappinglabeltoindex(label,mapping);
     label_index = single(label_index) + 1;
     mask_context = zeros(img_size_h,img_size_w,'single');
     for ind = 1:2
        mask_context(label_index== context_class(ind)) = 1;
     end
     mask_context(label_index==invalid_class) = 1;
     proposal = reshape(proposal,[img_size_h*img_size_w,3]);
     for ind = 1:size(object_class,2)
        mask = zeros(img_size_h,img_size_w,'single');
        mask(label_index== object_class(ind)) = 1;
        connected_component = bwlabel(mask);
        conn = unique(connected_component);  
       
          for k = 1:size(conn,1)
              [row, col] = find(connected_component==conn(k));
             if(mask(row(1),col(1))==1)
                mask_box = zeros(img_size_h,img_size_w);
                %mask_box_tight = mask_box;
                
                mask_answer = mask_box;
                mask_answer(sub2ind([img_size_h,img_size_w],row,col)) = 1;
                min_r = min(row); min_c = min(col);
                max_r = max(row); max_c = max(col);
                length_r = max_r-min_r+1; length_c = max_c-min_c+1;
                %start_r = max(round(min_r-length_r*width), 1); 
                start_r = min_r;
                start_c = max(round(min_c-length_c*extension_width),1);
                end_r = min(round(max_r+length_r*extension_width),img_size_h); end_c = min(round(max_c+length_r*extension_width),img_size_w);
              
                mask_box(start_r:end_r,start_c:end_c) = 1;
               % mask_box_tight(min_r:max_r,min_c:max_c)= 1;
               
                
                boundary_answer = edge(mask_answer,'Sobel',0);
                boundary_answer = imdilate(single(boundary_answer), se_outer);
              
                
                index = find(mask_context==1&mask_box==1&boundary_answer==1);
                
                proposal(index,1) = 0;
                proposal(index,2) = 0;
                proposal(index,3) = 0;
            end
          end
     end
     proposal = reshape(proposal,[img_size_h,img_size_w,3]);
     label = reshape(label,[img_size_h,img_size_w,3]);
      
 if visulize
        figure(1);
        imshow(proposal);
        figure(2);
        imshow(uint8(label));
      %  figure(3);
     %   imshow(img);
        pause;
 end
     label = uint8(label);
     imwrite(uint8(proposal*255.0),[save_response_img list{1,i}(1:end-4) '.jpg']);
    % imwrite(uint8(label),[save_revised_label list{1,i}(1:end-4) '.png']);
     save_warper2([save_response_path,list{1,i}(1:end-4) '.mat'],proposal,answer,label);
end