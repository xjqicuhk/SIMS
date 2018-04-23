function [a] = save_warper_3(string, proposal,mask,class,proposal_iou,proposal_pole_mask,response_index,original_index,mask_remained,class_remained,original_index_remained,all_class_flag)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

 a=0;
 save(string,'proposal','mask','class','proposal_iou','proposal_pole_mask','response_index','original_index','mask_remained','class_remained','original_index_remained','all_class_flag');
end

