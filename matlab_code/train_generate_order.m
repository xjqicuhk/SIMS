% cityscape provide disparity map, other datasets directly provide depth
% map: disparity larger, means object closer
% we decide the relative order of two adjacent segments by checking their
% boundary part
label_path = '../traindata/label_refine/';
img_path = '../traindata/RGB256Full/';
disparity_path = '../traindata/disparity/';
mask_path = '../traindata/original_segment/';

list = dir([label_path '*.png']);
list = struct2cell(list);


order_list = load('order.mat');
order_list = order_list.order;

%% optional for other datasets
% cases depth failed objects interacy with road
% sanity check
% objects vs planaer regions(optional)
% provide by the cityscape dataset
objects = [6,7,8,15,16,17,14,18,19,12,13];
planer = [1,2,10];
%%
se = strel('square', 3);
se2 = strel('square', 11);
img_size_h = 256;
img_size_w = 512;
save_path = '../traindata/order/';
mkdir(save_path);
load('cityscapes_colormap.mat');
load('mapping.mat');
frequency = zeros(19,1);
visualize = 0;
%% 
for i =1:2975
    i
    label = imread([label_path, list{1,i}]);
    label = func_mappinglabeltoindex(label,mapping);
    label = single(label);
    label = label+1;
    img = imread([img_path list{1,i}]);
    disparity = imread([disparity_path list{1,i}]);
    
    semantic_segment_mask = zeros(img_size_h,img_size_w,3,5,'uint8');
    semantic_segment_label = zeros(5,1);
    count =1;
    
    for c = 1:size(order_list,2)
        library = load([mask_path sprintf('%02d/',order_list(c)) list{1,i}(1:end-4) '.mat']);
        library_mask = library.library_mask;
        for c2 = c+1:size(order_list,2)
        %% checking the relative order of two adjacent segments 
        %  by comparing the median value in the dilated boundary
            library_2 = load([mask_path sprintf('%02d/',order_list(c2)) list{1,i}(1:end-4) '.mat']);
            library_2_mask = library_2.library_mask;
            for j = 1:size(library_mask,3)
                if(sum(sum(library_mask(:,:,j)))>1000)
                for k=1:size(library_2_mask,3)
                    if(sum(sum(library_2_mask(:,:,k)))>1000)
                    edge1 =   edge(library_mask(:,:,j),'Sobel',0);
                    edge1 = imdilate(single(edge1), se);
                    mask_all = edge1 + single(library_mask(:,:,j));
                    index = find(mask_all==1&library_2_mask(:,:,k)==1);
                    if(~isempty(index))
                        edge1 = edge(library_mask(:,:,j),'Sobel',0);
                        edge1 = imdilate(single(edge1), se2);
                        edge2 = edge(library_2_mask(:,:,k),'Sobel',0);
                        edge2 = imdilate(single(edge2), se2);
                        count_region = zeros(img_size_h,img_size_w,'single');
                        count_region(edge1==1&edge2==1) = 1;
                        region1_disp = single(disparity).*single(library_mask(:,:,j)).*count_region;
                        region2_disp = single(disparity).*single(library_2_mask(:,:,k)).*count_region;
                        val1 = region1_disp(region1_disp>0&(~isnan(region1_disp)));
                        val2 = region2_disp(region2_disp>0&(~isnan(region2_disp)));
                        val1_median = median(val1);
                        val2_median = median(val2);
                        mask_all = single(library_mask(:,:,j)) + single(library_2_mask(:,:,k));
                       
                        label_mask = single(label).*(mask_all);
                        label_mask(label_mask==0) = 256;
                        label_mask = uint8(label_mask-1);
                        
                        semantic_segment_mask(:,:,:,count) = uint8(ind2rgb(label_mask,colormap)*255.0);
                        if((~isnan(val1_median))&(~isnan(val2_median)))
                        if(val1_median<=val2_median)
                            semantic_segment_label(count) = order_list(c2);
                            frequency(order_list(c)) = frequency(order_list(c))+1;
                        elseif(val1_median>val2_median)
                            semantic_segment_label(count) = order_list(c);
                            frequency(order_list(c2)) = frequency(order_list(c2))+1;
                        end
                        %% comment the following if do not need planert object checking
                        % begin comment
                        if(ismember(order_list(c),planer)&ismember(order_list(c2),objects))
                            semantic_segment_label(count) = order_list(c2);
                            frequency(order_list(c)) = frequency(order_list(c))+1;
                        end
                       % end comment      
                 %% visualize  ordering      
                      if(visualize)
                        figure(1);
                        imshow(label,colormap);
                        figure(2);
                        imshow(library_mask(:,:,j),[]);
                        figure(3);
                        imshow(library_2_mask(:,:,k),[]);
                        figure(4);
                        imshow(semantic_segment_mask(:,:,:,count));
                        semantic_segment_label(count)
                        figure(5);
                        imshow(img);
                         pause;
                      end
                   
                        count = count +1;
                       
                        end
                    end
                    end
                end
                end
            end
            
        end
        
    end
    semantic_segment_mask = semantic_segment_mask(:,:,:,1:count-1);
    semantic_segment_label = semantic_segment_label(1:count-1,:);
    
    save([save_path list{1,i}(1:end-4) '.mat'],'semantic_segment_mask','semantic_segment_label');
    
end
%% overall statistics for coarse order generation
save('frequency.mat','frequency');
