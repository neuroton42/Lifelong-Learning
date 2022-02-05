%% Weak regression enhanced lifelong learning
clear all;

dataset=1;
%%
if dataset==1
%% school data

data=struct2cell(importdata('school.mat'));
x_cell=data(1,:); x_cell=x_cell{1,1};
y_cell=data(2,:); y_cell=y_cell{1,1};
T_max=size(y_cell,2);

for i=1:T_max
    task_idx(i)=size(x_cell{i},1);
    task_idxsum(i)=sum(task_idx(1:i));
    if i==1
        b=1;
    else
        b=task_idxsum(i-1)+1;
    end
    x_raw(b:task_idxsum(i),:)=x_cell{i}; 
    y_raw(b:task_idxsum(i),:)=y_cell{i};   
end

x=x_raw';
y=y_raw';
%}
elseif dataset==2
%% PD1
A=importdata('park.txt');
%te=570; % 570 1180 1970
sub=(A(1:end,1)); % 1 index
data=(A(1:end,[5,7:22])); % 5 motor_UPDRS/output, 7-22, input  

%{
for t=1:17
    figure(t)
    plot(data(:,t))
end
%}
j=0;
k=1;

x=data(:,2:end)';
y=data(:,1)';

num=length(sub); % total number of data points
for i=1:num
    if i<=num-1
        if sub(i)==sub(i+1)
     j=j+1;
  else
      task_idx(k)=j+1;
      k=k+1;
      j=0;
  end
    else
      task_idx(k)=j+1; % single index
    end
end
T_max=size(task_idx,2); % number of tasks
%}
elseif dataset==3
%% PD2
A=importdata('park.txt');
%te=570; % 570 1180 1970
sub=(A(1:end,1)); % 1 index
data=(A(1:end,[6,7:22])); % 6 Parkinson-Total/output, 7-22, input  
j=0;
k=1;

x=data(:,2:end)';
y=data(:,1)';

num=length(sub); % total number of data points
for i=1:num
    if i<=num-1
        if sub(i)==sub(i+1)
     j=j+1;
  else
      task_idx(k)=j+1;
      k=k+1;
      j=0;
  end
    else
      task_idx(k)=j+1; % single index
    end
end
T_max=size(task_idx,2); % number of tasks
else 
%% AD
load ADAS_data
T_max=size(X,1);
for t=1:T_max
  task_idx(t)=size(X{t},1);  
end
x=cell2mat(X)';
y=cell2mat(Y)';
%}
end
%% normalization

for i=1:size(x,1)
[x(i,:),~]=mapminmax(x(i,:),0,1); % x:d*N y:1*N
end
x=x'; % x:N*d
[y,PS]=mapminmax(y,0,1);

%% train/test
for ii=1:1
K=T_max;
training_percent=0.5;
for i=1:K
    if i==1
    xc1{i}=x(1:task_idx(i),:);   
    yc1{i}=y(1:task_idx(i));
    else
    a=sum(task_idx(1:i-1));
    xc1{i}=x(a+1:a+task_idx(i),:);
    yc1{i}=y(a+1:a+task_idx(i));
    end
end  
     
task_num=[1:T_max];
for i=1:K    
    xc{i}=xc1{task_num(i)};
    yc{i}=yc1{task_num(i)};    
    x_tr{i}=xc{i}(1:round( size(xc{i},1)*training_percent ),:);
    y_tr{i}=yc{i}(1:round( size(xc{i},1)*training_percent ))';
    x_te{i}=xc{i}(round( size(xc{i},1)*training_percent )+1:end,:);
    y_te{i}=yc{i}(round( size(xc{i},1)*training_percent )+1:end)';
    %}
end

T_max=length(task_idx);

%% cross validation
% parameter range
% pd: k 1-6/ rho 1-5/ rho2 1-5/
% ad1: k 1-6/ rho 4-8/ rho2 4-8/
% ad2: k 1-6/ rho 4-8/ rho2 4-8/
%{
fold=5;
for i=1:K
    indices=crossvalind('Kfold',size(x_tr{i},1),fold);
    for j=1:fold
        test=(indices==j);
        train=~test;
        x_croesstr{i,j}=x_tr{i}(train,:);
        x_croesste{i,j}=x_tr{i}(test,:);
        y_croesstr{i,j}=y_tr{i}(train,:);
        y_croesste{i,j}=y_tr{i}(test,:);
    end
end

lat_var=2; 
cout=0;
for iter1=1:8 
k=iter1;
for iter2=1:8
rho=10^(-iter2); 
for iter3=1:8
rho2=10^(-iter3); 
cout=cout+1
for i=1:1
   xvtr=x_croesstr(:,i)'; 
   yvtr=y_croesstr(:,i)';
   xvte=x_croesste(:,i)'; 
   yvte=y_croesste(:,i)';
   [yp_tr,yp_te,ytr,yte,yp,ype] = CoupleDic_ELLA_WeakRe(x_tr,y_tr,x_te,y_te,k, rho,rho2, 0.1,1);
   rmse(i) = sqrt(mean((yte - ype).^2));
end
rmse2(iter1,iter2,iter3)=mean(rmse);
end
end
end
%[a1,a2,a3]=find(rmse2==min(min(rmse2)));

for i=1:size(rmse2,1)
for j=1:size(rmse2,2)
for k=1:size(rmse2,3)
if rmse2(i,j,k)==min(min(min(rmse2)))
best_para_idx=[i j k] 
end
end
end
end
%}
%% algorithm
% parameters are determined by CV
k=1;                     
rho=10^(-2);             
rho2=10^(-5);            
rou=0.1;  
[yp_tr,yp_te,ytr,yte,yp,ype,ACTpT] = CoupleDic_ELLA_WeakRe(x_tr,y_tr,x_te,y_te,k, rho,rho2, rou,1);
%[yp_tr,yp_te,ytr,yte,yp,ype,ACTpT] = CoupleDic_ELLA_WeakFea(x_tr,y_tr,x_te,y_te,k, rho,rho2, rou,1);

ytr1=mapminmax('reverse',ytr,PS); 
yte1=mapminmax('reverse',yte,PS); 
yp1=mapminmax('reverse',yp,PS); 
ype1=mapminmax('reverse',ype,PS); 
MSE_te(ii)=(mean((yte1 - ype1).^2))

end


