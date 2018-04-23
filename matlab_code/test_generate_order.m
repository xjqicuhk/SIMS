%path for label map
label_path = '../testdata/label_refine/';
% original segment masks
mask_path = '../testdata/original_mask/';

list = dir([label_path '*.png']);
list = struct2cell(list);

order_list = load('order.mat');
order_list = order_list.order;

img_size_h = 256;
img_size_w = 512;
save_path = '../testdata/order/';
mkdir(save_path);
load('cityscapes_colormap.mat');
load('mapping.mat');
max_count = 0;
max_num = 5;
for i =1:size(list,2)
    i
    
    label = imread([label_path, list{1,i}]);
    label = func_mappinglabeltoindex(label,mapping);
    label = single(label);
    label = label+1;
    
    semantic_segment_mask = zeros(img_size_h,img_size_w,3,max_num,'uint8');
    semantic_segment_label = zeros(max_num,4);
  
    count =1;
    
    for c = 1:size(order_list,2)
        library = load([mask_path sprintf('%02d/',order_list(c)) list{1,i}(1:end-4) '.mat']);
        library_mask = library.library_mask;
        
        
        for c2 = c+1:size(order_list,2)
            library_2 = load([mask_path sprintf('%02d/',order_list(c2)) list{1,i}(1:end-4) '.mat']);
            library_2_mask = library_2.library_mask;
            for j = 1:size(library_mask,3)
                if(sum(sum(library_mask(:,:,j)))>1000)
                for k=1:size(library_2_mask,3)
                    if(sum(sum(library_2_mask(:,:,k)))>1000)
                    %   edge1 =   edge(library_mask(:,:,j),'Sobel',0);
                    %   edge1 = imdilate(single(edge1), se);
                       mask_all = single(library_mask(:,:,j)) + single(library_2_mask(:,:,k));
                      % index = find(mask_all==1&library_2_mask(:,:,k)==1);
                    
                    
                        label_mask = single(label).*(mask_all);
                        label_mask(label_mask==0) = 256;
                        label_mask = uint8(label_mask-1);
                        
                        semantic_segment_mask(:,:,:,count) = uint8(ind2rgb(label_mask,colormap)*255.0);
                        semantic_segment_label(count,:) = [j,k,order_list(c), order_list(c2)];
                        count = count +1;
                       
                       
                    end
                  end
                end
            end
            
        end
        
    end
    semantic_segment_mask = semantic_segment_mask(:,:,:,1:count-1);
    semantic_segment_label = semantic_segment_label(1:count-1,:);
    if(max_count<count-1) max_count = count -1;end
    
    save([save_path list{1,i}(1:end-4) '.mat'],'semantic_segment_mask','semantic_segment_label');
    
end
max_count
