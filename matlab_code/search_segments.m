% database path traing data and train data segments (external memory for searching)
traindata_path = '../traindata/RGB256Full/';
traindata_segment_path = '../traindata/original_segment/';
list_train = dir([traindata_path '*.png']);
list_train = struct2cell(list_train);

% test data path
testdata_label_path = '../testdata/label_refine/';
list_test = dir([testdata_label_path '*.png']);
list_test = struct2cell(list_test);
testdata_segment_path = '../testdata/original_mask/';

% image resolution
img_size_h = 256;
img_size_w = 512;

% searched segement
save_response_path = '../data/testdata/searched_top10_segments/';
mkdir(save_response_path);

for i = 1:19
    mkdir([save_response_path sprintf('%02d',i) ,'/']);
end

topk = 10;


for i = 1:size(list_test,2)
    i
  for c = 1:19
  
    save_response_path_final = [save_response_path sprintf('%02d',c)];
  
    query = load([testdata_segment_path  sprintf('%02d/',c) list_test{1,i}(1:end-4) '.mat']);
    query_mask = query.library_mask;
    query_mask = single(query_mask);
     
    answer = zeros(img_size_h,img_size_w,3,size(query_mask,3),'single');
    
   
    proposal = zeros(img_size_h,img_size_w,3,topk,size(query_mask,3),'uint8');
    proposal_iou = zeros(topk,size(query_mask,3),'single');
    
    proposal_pole_mask = zeros(img_size_h,img_size_w,topk,size(query_mask,3),'uint8');
    response_index = zeros(topk,size(query_mask,3),'single');
   
    
   if(~exist([save_response_path_final '/' list_test{1,i}(1:end-4) '.mat'],'file'))
    for j = 1:size(query_mask,3)
      
        record_iou = zeros(topk,1,'single');
        record_query_context_iou = zeros(topk,1,'single');
       
        
        if(sum(sum(query_mask(:,:,j)))>1000)
            summation = -10.0;
            current_index = 0.0;
            
            for k = 1:size(list_train,2)
                [min_iou, min_ind]= min(record_iou(:));
                response = load([traindata_segment_path, sprintf('%02d/',c),list_train{1,k}(1:end-4) '.mat']);
                response_mask = response.library_mask;
                response_img = response.library_img;
                response_pole_mask = response.library_mask_pole;
               for m = 1:size(response_mask,3)
                   
                  iou = sum(sum(and(response_mask(:,:,m),query_mask(:,:,j))))/sum(sum(or(response_mask(:,:,m),query_mask(:,:,j))));
                   if(iou>min_iou)
                    
                      record_iou(min_ind) = iou;
                      
                      proposal(:,:,:,min_ind,j) = (response_img(:,:,:,m));
                      proposal_pole_mask(:,:,min_ind,j) = response_pole_mask(:,:,m);
                      response_index(min_ind,j) = k;
                  end
               end
            end
        end
      proposal_iou(:,j) = record_iou;
      
    end
    save([save_response_path_final '/' list_test{1,i}(1:end-4) '.mat'],'proposal','proposal_iou','proposal_pole_mask','response_index');
   end
    
  end

end