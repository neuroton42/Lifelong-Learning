function [yp_tr,yp_te,ytr,yte,yp,ype,ACTpT] = CoupleDic_ELLA_WeakFea(x_tr,y_tr,x_te,y_te,k, rho,rho2, rou,single_lifelong_sign)
%% weak prediction enahnced ELLA with coupled dictionairs learning 

[~,T_max]=size(x_tr);
[~,d]=size(x_tr{1});
for i=1:T_max
    idx(i)=size(y_te{i},1);
end
me_num1=d+1;
%me_num2=min(idx);
me_num2=6;
A1=zeros(k*me_num1,k*me_num1);
b1=zeros(k*me_num1,1);
A2=zeros(k*me_num2,k*me_num2);
b2=zeros(k*me_num2,1);
L=0.5*ones(me_num1,k);
D=0.5*ones(me_num2,k);
%L=rand(me_num1,k);
%D=rand(me_num2,k);
opts=[];
opts=init_opts(opts);
e=0;
T=0;
tic;
for t=1:T_max
    T=T+1;
    X=x_tr{t}';
    Y=y_tr{t};
    
    % model 
    %X_new=[ones(1,size(X,2));X]; theta1{t}=pinv(X_new*X_new')*X_new*Y; 
    [~,~,~,~,theta1{t}] = plsregress(X',Y,2);X_new=[ones(1,size(X,2));X];
    D1{t}=(X_new*X_new')/(2*size(X_new,2)); % hession matrix 1
    % weak prediction
    prect=[ones(size(x_te{t},1),1),x_te{t}];
    ys=prect*theta1{t};
    theta2{t}=[min(ys),max(ys),mean(ys),median(ys),std(ys),var(ys)]';
    D2{t}=rou*eye(me_num2);
    
    %{
    figure(t)
    plot(ys)
    hold on
    plot(y_te{t})
    %}
    % cor 
    %{
    R=[];
    for i=1:size(X,1)
        if all (~(diff(X(i,:))))
    R(i)=0;        
        else
    R(i) = corr(X(i,:)',Y,'type','pearson');
        end
    end
    theta2{t}=R';
    D2{t}=rou*eye(me_num2);
    %}
    
    % compute s
    betal{t}=[theta1{t};theta2{t}];
    A{t}=[D1{t},zeros(me_num1,me_num2);zeros(me_num2,me_num1),D2{t}];
    K=[L;D];
    [s{t}, funcVal] = my_Lasso(K, betal{t}, A{t}, rho, opts);
    
    % update L
    A1=A1+kron( (s{t}*s{t}')',D1{t});
    b1=b1+ ( kron( s{t}', theta1{t}'*D1{t} ) )';
    L_vec=inv((1/T)*A1+rho2*eye(k*me_num1,k*me_num1))*(1/T)*b1;
    L=reshape(L_vec,[me_num1,k]);
    
    % update D
    A2=A2+kron( (s{t}*s{t}')',D2{t});
    b2=b2+ ( kron( s{t}', theta2{t}'*D2{t} ) )';
    D_vec=inv((1/T)*A2+rho2*eye(k*me_num2,k*me_num2))*(1/T)*b2;
    D=reshape(D_vec,[me_num2,k]);
    
    %{
    for i=1:T
        e=e+((betal{i}-K*s{i})'*A{i}*(betal{i}-K*s{i}))+rho*norm(s{i},1);
    end
    g2(t)=(1/T)*e+rho2*norm(K,'fro');
    e=0;
    %}
end
time=toc;
%plot(g2)
ACTpT=time*10^3/T_max;
s=cell2mat(s);
if single_lifelong_sign==1
theta=L*s; % lifelong learning
else 
theta=cell2mat(theta1); % single-task
end
%% training/test result
ytr=[];yp=[];
for t=1:T_max
    % training result
    yp_tr{t}=[ones(size(x_tr{t},1),1),x_tr{t}]*theta(:,t);
    %yp_tr{t}=[x_tr{t}]*theta(:,t);
    task_tr_idx(t)=size(x_tr{t},1);
    task_tr_idxsum(t)=sum(task_tr_idx(1:t));
    if t==1
        b=1;
    else
        b=task_tr_idxsum(t-1)+1;
    end
    ytr(b:task_tr_idxsum(t))=y_tr{t}; 
    yp(b:task_tr_idxsum(t))=yp_tr{t}; 
    
    yp_te{t}=[ones(size(x_te{t},1),1),x_te{t}]*theta(:,t);
    %yp_te{t}=[x_te{t}]*theta(:,t);
    task_te_idx(t)=size(x_te{t},1);
    task_te_idxsum(t)=sum(task_te_idx(1:t));
    if t==1
        a=1;
    else
        a=task_te_idxsum(t-1)+1;
    end
    yte(a:task_te_idxsum(t))=y_te{t}; 
    ype(a:task_te_idxsum(t))=yp_te{t}; 
   
end


end