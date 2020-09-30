function [W,Omega]=MTERL(data,label,lambda,gamma,nu)
%the Omega sloved in the demo can  refer to: 
%Yu Zhang and Dit-Yan Yeung. A Convex Formulation for Learning Task Relationships in Multi-Task Learning. 
%In: Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence (UAI), 2010.

m=length(label); 
d=size(data{1,1},2);
W=zeros(d,m); 
V=zeros(d,m);

[W,Omega]=myAPG(W,V,data,label,lambda,gamma,nu);

end
function [W,Omega]=myAPG(W_initial,V,data,label,lambda,gamma,nu)
m=length(label);
d=size(data{1,1},2);
epsilon=10^(-8);
max_iteration=1000;
t=0;  alpha = 1; W=W_initial;
Omega=eye(m)/m;
[data_O,label_O,task_index,ins_num]=PreprocessMTData(data,label);

n=size(data_O,1);
insIndex=cell(1,m);
ins_indicator=zeros(m,n);
for i=1:m
    insIndex{i}=sort(find(task_index==i));
    ins_indicator(i,insIndex{i})=1;
end
threshold=10^(-12);
model.alpha=zeros(1,n);
model.b=zeros(1,m);kertype='linear';kerpar=0;
Km=CalculateKernelMatrix(data_O,kertype,kerpar);
m_Cor=real(Omega/(lambda*Omega+gamma*eye(m)));

for iter=1:max_iteration
    old_model=model;
    old_Omega=Omega;
    MTKm=Km.*m_Cor(task_index,task_index);
    model=MTRL_RR(MTKm,label_O,task_index,insIndex,ins_num);
    clear MTKm;
    temp=m_Cor(:,task_index)*diag(model.alpha);
    temp=temp*Km*temp';
    [eigVector,eigValue]=eig(temp+epsilon*eye(m));
    clear temp;
    eigValue=sqrt(abs(diag(eigValue)));
    eigValue=eigValue/sum(eigValue);
    Omega=eigVector*diag(eigValue)*eigVector';
    m_Cor=eigVector*diag(eigValue./(lambda*eigValue+gamma))*eigVector';
    clear eigVector eigValue;
    if norm(model.alpha-old_model.alpha,2)<=threshold*n&&norm(model.b-old_model.b,2)<=threshold*m&&norm(Omega-old_Omega,'fro')<=threshold*m*m
        clear old_model old_Omega;
        break;
    end
    clear old_model old_Omega; 
    U=(1-alpha)*W+alpha*V;
    Lu=100000;%should be set
    G=[];
    for i=1:d
        Vi=min(1,max(-1,U(i,:)/nu));
        G=[G;sum(abs(U(i,:)))*Vi];       
    end
    tmp_o = U*Omega^(-1);
    for i=1:m
        V(:,i)=V(:,i)-1/(alpha*Lu)*( data{1,i}'* (data{1,i}* U(:,i)-label{1,i}') + lambda*G(:,i) +gamma* tmp_o(:,i));
    end  
    W=(1-alpha)*W+alpha*V;
    alpha=2/(t+1);
    t=t+1;    
    F_wh=0;  
    for i=1:m
        F_wh = F_wh+norm(label{1,i}-W(:,i)'*data{1,i}',2)^2/(2*m);
    end
    for i=1:d      
        F_wh =F_wh+lambda*sum( abs(W(i,:)) )^2/2;
    end
    F_wh = F_wh + gamma * trace(W*Omega^(-1)*W')/2;
    fValue(iter,1)=F_wh;    
    if (iter>10 && (abs(fValue(iter,1)-fValue(iter-1,1))/abs(fValue(iter-1,1))<epsilon))   
        break;
    end   
end
end