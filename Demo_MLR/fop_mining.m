function [model] = fop_mining(net_mat,label)
type=4; 
level=3; 
fold_num = 1; 
parameter_num = 12;

if(type==4)
[node_num,~,net_num]=size(net_mat);
per_fold = round(net_num/fold_num);
pos_freq_cell = cell(fold_num, 1);
pos_subgraph_cell = cell(fold_num, 1);
neg_freq_cell = cell(fold_num, 1);
neg_subgraph_cell = cell(fold_num, 1);

pos_thres=zeros(level,1);
step=1.5;
pos_thres(1)=0.5*step;
for ti=2:level
   pos_thres(ti)=pos_thres(ti-1)*(step+0.1)*0.5;
end
pos_down_bound=0.03; 
pos_up_bound = 0.35;
    

neg_thres=zeros(level,1);
step=1.5;
neg_thres(1)=0.5*step;
for ti=2:level
   neg_thres(ti)=neg_thres(ti-1)*(step+0.1)*0.5;
end
neg_down_bound=0.03; 
neg_up_bound = 0.35;
end
disp('start mining.');

%%
for fi=1:fold_num
    disp(['mining the ',num2str(fi),'-th fold''s subgraphs'])
    if fold_num ~= 1
        max_idx = fi * per_fold;
        if fi==fold_num
            max_idx = net_num;
        end
        min_idx = (fi - 1) * per_fold + 1;
        tmp_label = label;
        tmp_label(min_idx:max_idx) = 0;
    else
       tmp_label = label; 
    end
    
    st=clock;
    
    [ neg_subgraph,neg_freq ] = frequent_weight_subgraph_mining( net_mat(:,:,tmp_label == -1),level,neg_down_bound,neg_thres);
 
    neg_subgraph_cell{fi} = neg_subgraph;
    neg_freq_cell{fi} = neg_freq;
    et=clock;
    disp(['The time of mining ',num2str(fi),'-th fold negative subgraphs is ',num2str(etime(et,st))]);
    
    st=clock;
    [ pos_subgraph,pos_freq ] = frequent_weight_subgraph_mining( net_mat(:,:,tmp_label == 1),level,pos_down_bound,pos_thres);
    pos_subgraph_cell{fi}=pos_subgraph;
    pos_freq_cell{fi} = pos_freq;
    et=clock;
    disp(['The time of mining ',num2str(fi),'-th fold positive subgraphs is ',num2str(etime(et,st))]);
end

model = cell(parameter_num,1);
model{1} = pos_subgraph_cell;
model{2} = pos_freq_cell;
model{3} = neg_subgraph_cell;
model{4} = neg_freq_cell;
model{5} = pos_thres;
model{6} = neg_thres;
model{7} = level;
model{8} = label;
model{9} = pos_up_bound;
model{10} = pos_down_bound;
model{11} = neg_up_bound;
model{12} = neg_down_bound;

end


function [ subgraph_cell,score_cell ] = frequent_weight_subgraph_mining( net_mat,level,alpha,thres)
[node_num,~,net_num]=size(net_mat);

edge_num=node_num*(node_num-1)/2;

global  subgraph_cell;
global  score_cell;
subgraph_cell=cell(level,1);
score_cell=cell(level,1);

edge2edge_weight(edge_num,edge_num,net_num)=false;
edge_idx1=1;

tic
for ni=1:node_num-1
    for nj=ni+1:node_num
        edge_idx2=1;
        for ni2=1:node_num-1
            for nj2=ni2+1:node_num                                                                           
                edge2edge_weight(edge_idx1,edge_idx2,:)=(net_mat(ni,nj,:)-net_mat(ni2,nj2,:)-alpha)>0;
                edge_idx2=edge_idx2+1;
            end
        end
        edge_idx1=edge_idx1+1;
    end
end
toc

%%
edge_idx_mat=zeros(node_num,node_num); 
idx=1;
for ni=1:node_num-1
    for nj=ni+1:node_num
        edge_idx_mat(ni,nj)=idx;
        idx=idx+1;
    end
end
edge_idx_mat=edge_idx_mat+edge_idx_mat';

%%
tic
for ni=1:node_num-1
    for nj=ni+1:node_num
     subgraph=[ni,nj];
     %disp('subgraph [' 1,2]');
     do_subgraph_mining(subgraph, node_num,net_num,edge2edge_weight,edge_idx_mat,level,thres);
     disp([ni,nj]);
     toc
    end
end
end

function do_subgraph_mining(subgraph,node_num,net_num,edge2edge_weight,edge_idx_mat,level,thres)

global  subgraph_cell;
global  score_cell;
contain_edge_num=size(subgraph,1);

if(contain_edge_num-1>=level)
    return;
end
contain_node=unique(subgraph);
contain_node_num=length(contain_node);

score= true(1,net_num);
for ei=1:contain_edge_num-1
    score=score & reshape(edge2edge_weight(edge_idx_mat(subgraph(ei,1),subgraph(ei,2)),edge_idx_mat(subgraph(ei+1,1),subgraph(ei+1,2)),:),1,net_num);
end
new_subgraph=[subgraph;0,0];

for ni=1:contain_node_num
   for nj=1:node_num

      if(nj==contain_node(ni))
          continue;
      end
      
      is_contain=false;
      for ei=1:contain_edge_num
         if (sum([contain_node(ni),nj]==subgraph(ei,:))==2 || sum([nj,contain_node(ni)]==subgraph(ei,:))==2) 
            is_contain=true;
            break;
         end
      end
      if(is_contain)
          continue;
      end
      
      new_score=score & reshape(edge2edge_weight(edge_idx_mat(subgraph(contain_edge_num,1),subgraph(contain_edge_num,2)),edge_idx_mat(contain_node(ni),nj),:),1,net_num);
      new_score=sum(new_score)/net_num;

      if(new_score>thres(contain_edge_num))
          new_subgraph(contain_edge_num+1,:)=[contain_node(ni),nj];
          subgraph_cell{contain_edge_num}{length(subgraph_cell{contain_edge_num})+1}=new_subgraph;
          score_cell{contain_edge_num}(length(score_cell{contain_edge_num})+1)=new_score;
          if(contain_edge_num<level)
             do_subgraph_mining(new_subgraph, node_num,net_num,edge2edge_weight,edge_idx_mat,level,thres);
          end
      end
   end
end
end

