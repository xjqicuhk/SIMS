img_path = '../data/RGB256Full/';

label_coarse_path = '../data/Label256Full_coarse/';
label_index_path = '../data/Label256Full_index/';

list = dir([img_path '*.png']);
list = struct2cell(list);

path_coarse_library = '../data/library_all_wo_pole_pred_refine_new/';

path_response_proposal = '../testdata/response_proposal_all_wo_pole_new_pred_refine_context_iou/';

save_path = '../testdata/transform/transform_256/';
mkdir(save_path);

img_size_h = 256;
img_size_w = 512;

%order_list = [11,3,4,5,9,10,1,2,6,7,8,15,16,17,14,18,19,12,13];
order_list = load('order.mat');
order_list = order_list.order;
max_num = 30;
load('mapping.mat');
% last 500 images is used for testing.
for i = 2976:size(list,2)
    i
    proposal = zeros(max_num,img_size_h,img_size_w,3,'uint8');
    mask = zeros(max_num,img_size_h,img_size_w,'uint8');
    class = zeros(max_num,1,'uint8');
    proposal_iou = zeros(max_num,1);
    proposal_pole_mask = zeros(max_num,img_size_h,img_size_w,'uint8');
   
    response_index = zeros(max_num,1,'uint32');
    original_index = zeros(max_num,1,'uint32');
    count = 0;
    count_remained = 0;
    count_all = 0;
   % proposal_remained = zeros(max_num,img_size_h,img_size_w,3,'uint8');
    mask_remained = zeros(max_num,img_size_h,img_size_w,'uint8');
    class_remained = zeros(max_num,1,'uint32');
    original_index_remained =  zeros(max_num,1,'uint32');
    all_class_flag = zeros(max_num,2,'uint32');
  for c_ind = 1:19
     c = order_list(c_ind);
     
    
     query = load_warper([path_coarse_library sprintf('%02d/',c) list{1,i}(1:end-4) '.mat']);
     query_mask = query.library_mask;
     query_img = query.library_img;

     query_mask = single(query_mask);
     
     data = load([path_response_proposal sprintf('%02d/',c) list{1,i}(1:end-4) '.mat']);
     answer = data.proposal;
     answer_iou = data.proposal_iou_all;
     answer_pole_mask = data.proposal_pole_mask;
     response_index_data = data.response_index;
    
   
    for j = 1:size(answer,5)
        count_all = count_all +1;
        iou_score = answer_iou(:,j);
        [max_score, max_ind] = sort(iou_score,'descend');
        
        t = answer(:,:,:,max_ind(1),j);
        t = squeeze(t);
        t = sum(t,3);
        t(t>0) = 1;
        t = imresize(t,[img_size_h,img_size_w],'nearest');
     
        if(sum(t(:))>0&&sum(sum(query_mask(:,:,j)))>1000)
            count  = count +1 ;
            
            answer_upsampled_img =imread([img_path list{1,response_index_data(max_ind(1),j)}(1:end-4) '.png']);
            answer_upsampled_img = im2double(answer_upsampled_img).*repmat(t,[1,1,3]);
            proposal(count,:,:,:) = uint8(answer_upsampled_img*255.0);
            mask(count,:,:) = imresize(query_mask(:,:,j),[img_size_h,img_size_w],'nearest');
            class(count) = c;
            proposal_iou(count) = max_score(1);
            tmp = imresize(single(answer_pole_mask(:,:,max_ind(1),j)),[img_size_h,img_size_w],'nearest');
            proposal_pole_mask(count,:,:) = squeeze(uint8(tmp));
            response_index(count) = response_index_data(max_ind(1),j);
            original_index(count) = j;
            all_class_flag(count_all,1) = 1;
            all_class_flag(count_all,2) = count;
        else 
           count_remained = count_remained + 1;
           mask_remained(count_remained,:,:) = imresize(query_mask(:,:,j),[img_size_h,img_size_w],'nearest');
           class_remained(count_remained) = c;
           original_index_remained(count_remained) = j;
           all_class_flag(count_all,1) = 2;
           all_class_flag(count_all,2) = count_remained;
        end
    end   
  end
  %the proposals in the image
  proposal = proposal(1:count,:,:,:);
  % mask 
  mask = mask(1:count,:,:);
  %class
  class = class(1:count);
  %proposal and query iou
  proposal_iou = proposal_iou(1:count);
  % trick about pole in cityscapes
  proposal_pole_mask = proposal_pole_mask(1:count,:,:);
  % proposal source image index
  response_index = response_index(1:count);
  % query original index
  original_index = original_index(1:count);
  
  mask_remained = mask_remained(1:count_remained,:,:);
  class_remained = class_remained(1:count_remained);
  original_index_remained = original_index_remained(1:count_remained);
  all_class_flag = all_class_flag(1:count_all,:);
  if(size(class_remained,1)+size(class,1)~=size(all_class_flag))
      print('error');
      break;
  end
  save_warper_3([save_path list{1,i}(1:end-4) '.mat'],proposal,mask,class,proposal_iou,proposal_pole_mask,...
      response_index,original_index,mask_remained,class_remained,original_index_remained,all_class_flag);
%    for ll = 1:size(proposal,1)
%     figure(1);
%     imshow(squeeze(proposal(ll,:,:,:)));
%     figure(2);
%     imshow(squeeze(mask(ll,:,:)),[]);
%     pause;
%    end
end