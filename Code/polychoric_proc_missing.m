function [MCORRPOLY]=polychoric_proc_missing(MSCORES,missingcode)                 
% Code developed by Luis Eduardo Garrido, Ph.D.
% e-mail: garrido.luiseduardo@gmail.com
% This code is based on the two stage procedure developed in 
% Olsson, U. (1979). Maximum likelihood estimation of the polychoric
% correlation coefficient. Psychometrika, 44(4), 443-460.
% INPUT PARAMETERS
% MSCORES: the MATLAB workspace should contain a data matrix named MSCORES 
% with the discrete scores on the variables of interest.
% missingcode: a discrete score used to identify the missing values in the
% MSCORES matrix. Even if there are no missing values a missingcode should
% be given.
% RESULTS
% Computes a polychoric correlation matrix named MCORRPOLY.
% NOTES
% Missing values are handled using pairwise deletion in order to compute 
% the polychoric correlations.  
% In the case of tetrachoric correlations with an observed cell frequency 
% of 0, the 0 is replaced with a value of 0.50 and the other cell 
% frequencies are adjusted to maintain the original marginal totals 
% according to the recommendations of Brown and Benedetti (1977) and Greer,
% Dunlap and Beatty (2003).
% Brown, M. B., & Benedetti, J. K. (1977). On the mean and variance of the 
% tetrachoric correlation coefficient. Psychometrika, 42, 347-355.
% Greer, T., Dunlap, W. P., & Beatty, G. O. (2003). A Monte Carlo
% evaluation of the tetrachoric correlation coefficient. Educational and
% Psychological Measurement, 63(6), 931-950.
% EXAMPLE
% [MCORRPOLY]=polychoric_proc_missing(MSCORES,999)  
% This syntax indicates that the matrix of discrete scores is named MSCORES
% and that the missing values are coded as 999. 
% END OF INSTRUCTIONS
missingcodenew=missingcode;
Size=size(MSCORES);
N=Size(1,1);                                                                
Var=Size(1,2);                                                              
critconv=1e-3;                                                             
NumIteraciones_N=1e6;                                                        
NumIteraciones_B=1e6;                                                        
VP=0;                                                                       
U=unique(MSCORES);                                                          
Size=size(U);                                                               
numU=Size(1,1);                                         
for I=1:numU
    if U(I)==missingcode                                                        
        VP=1;
    end
end
if VP==1
    missingcodenew=min(min(MSCORES))-1;                                         
    for J=1:Var
        for I=1:N
            if MSCORES(I,J)==missingcode
                MSCORES(I,J)=missingcodenew;
            end
        end
    end
end
U=unique(MSCORES); 
if VP==1
    for J=1:Var                                                             
        Size=size(U);
        numU=Size(1,1);
    for I=1:N
        K=1; cambio=0;
        while K<=numU && cambio==0
            if MSCORES(I,J)==U(K)
                MSCORES(I,J)=K-1;
                cambio=1;
            end
            K=K+1;
        end
    end
    end 
else
    for J=1:Var                                                             
        Size=size(U);
        numU=Size(1,1);
        for I=1:N
            K=1; cambio=0;
            while K<=numU && cambio==0
                if MSCORES(I,J)==U(K)
                    MSCORES(I,J)=K;
                    cambio=1;
                end
                K=K+1;
            end
         end
    end 
end
if VP==0
    Vpuntmax=max(MSCORES);                                                  
    puntmax=max(max(MSCORES));                                              
else
    missingcodenew=min(min(MSCORES));
    puntmax=max(max(MSCORES));
    Vpuntmax=max(MSCORES);                                                  
    Vpuntmax=min(Vpuntmax,puntmax);
end
MFREC=zeros(puntmax,Var);                                                   
for J=1:Var                                                                 
    for I=1:N
        if MSCORES(I,J)~=missingcodenew                                         
            Val=MSCORES(I,J);
            MFREC(Val,J)=MFREC(Val,J)+1;        
        end
    end    
end
if VP==0
    MPROB=MFREC/N;                                                          
else
    MPROB=zeros(puntmax,Var);                                               
    for J=1:Var
        Nnew=N;                                                             
        for I=1:N
            if MSCORES(I,J)==missingcodenew
                Nnew=Nnew-1;
            end
        end
        MPROB(:,J)=MFREC(:,J)/Nnew;
    end
end
    
MPROBAC=zeros(puntmax,Var);                                                 
for J=1:Var                                                                 
    L=Vpuntmax(J);
    for I=1:L
        for K=1:I
            MPROBAC(I,J)=MPROBAC(I,J)+MPROB(K,J);
        end        
    end      
end
MUMB=zeros(puntmax-1,Var);                                                  
for J=1:Var                                                                 
    for I=1:Vpuntmax(J)-1
        umbral=-2^0.5*erfcinv(2*MPROBAC(I,J));                              
        if umbral==-inf
            umbral=-10;
        end
        MUMB(I,J)=umbral;        
    end    
end
MUMBEX=zeros(puntmax+1,Var);                                                
MUMBEX(2:puntmax,:)=MUMB;                                                   
MUMBEX(1,:)=-10;                                                            
for J=1:Var
    L=Vpuntmax(J)-1;                                                        
    MUMBEX(L+2,J)=10;                                                       
end
MPUNTRED=zeros(N,2);                                                        
MCORRPOLY=eye(Var,Var);                                                     
numcorr=(Var^2-Var)/2;                                                      
MCOMBVAR=zeros(numcorr,2);
K=1;
for I=1:Var
    for J=I+1:Var
        MCOMBVAR(K,1)=I;                                                    
        MCOMBVAR(K,2)=J;                                                    
        K=K+1;
    end
end 
MUMBEXRED=zeros(puntmax+1,2);                                               
CORR=eye(2,2);                                                              
AA= [0.3253030 0.4211071 0.1334425 0.006374323];                            
BB= [0.1337764 0.6243247 1.3425378 2.2626645];     
a1= 0.319381530;                                                            
a2=-0.356563782;
a3= 1.781477937;
a4=-1.821255978;
a5= 1.330274429;
for KK=1:numcorr                                                             
    Iteracion=0; Iteracion1=0;
    II=MCOMBVAR(KK,1);
    JJ=MCOMBVAR(KK,2);
    MPUNTRED(:,1)=MSCORES(:,II);      
    MPUNTRED(:,2)=MSCORES(:,JJ);
    puntmax1=Vpuntmax(II);                                                  
    puntmax2=Vpuntmax(JJ);                                                  
    MUMBEXRED(1:puntmax1+1,1)=MUMBEX(1:puntmax1+1,II);    
    MUMBEXRED(1:puntmax2+1,2)=MUMBEX(1:puntmax2+1,JJ);     
    combumb=puntmax1*puntmax2*4;                                            
    Vector=zeros(combumb,1);
    X=zeros(combumb,2);  
    X2=zeros((puntmax1+1)*(puntmax2+1),5);
    TCONT=zeros(puntmax1,puntmax2);                                             
    for I=1:N
        if MPUNTRED(I,1)~=missingcodenew && MPUNTRED(I,2)~=missingcodenew  
            A=MPUNTRED(I,1);
            B=MPUNTRED(I,2);
            TCONT(A,B)=TCONT(A,B)+1;        
        end
    end    
    CatRes=size(unique(MPUNTRED));
    CatRes=CatRes(1,1); 
    Cambio=0;
    if CatRes==2                                                            
        for I=1:2
            for J=1:2
                if TCONT(I,J)==0
                    TCONT(I,J)=0.50;                                        
                    TCONT(J,I)=TCONT(J,I)+0.50;
                    Cambio=1;
                end
            end
        end
        if Cambio==1
            for I=1:2
                for J=1:2                
                    if TCONT(I,J)==round(TCONT(I,J))
                        TCONT(I,J)=TCONT(I,J)-0.5;
                    end
                end
            end
        end
    end
    K=1;  
    for I=2:puntmax1+1    
        for J=2:puntmax2+1
            X1=[MUMBEXRED(I,1)   MUMBEXRED(J,2);                            
                MUMBEXRED(I-1,1) MUMBEXRED(J,2); 
                MUMBEXRED(I,1)   MUMBEXRED(J-1,2); 
                MUMBEXRED(I-1,1) MUMBEXRED(J-1,2)];
            X(K:K+3,:)=X1;
            K=K+4; 
        end  
    end
    K=1;
    for I=1:puntmax1+1
        for J=1:puntmax2+1
            X2(K,1)=MUMBEXRED(I,1);                                         
            X2(K,2)=MUMBEXRED(J,2);
            K=K+1;
        end
    end
    MCORR=corrcoef(MPUNTRED);
    corr=MCORR(2,1);
    if corr==1 || corr==-1
        if corr==1
            corr=0.999;
        else
            corr=-0.999;
        end
    end     
    CORR(1,2)=corr; CORR(2,1)=corr; corrdif=1;                              
    corr_final=0; corrinf=-1; corrsup=1; Iteracion2=1;                      
    while corrdif>critconv    
        PDF=zeros(combumb,1);                                               
        GUV=zeros(combumb,1);                                                
        CDF=zeros(combumb,1);                                               
        for K=1:(puntmax1+1)*(puntmax2+1)
            X2(K,3)=1/(2*3.14159265*(1-corr^2)^0.5)*exp(-(X2(K,1)^2-2*corr*X2(K,1)*X2(K,2)+...
                   X2(K,2)^2)/(2*(1-corr^2)));           
            X2(K,4)=X2(K,3)*(X2(K,1)*X2(K,2)*(1-corr^2)+corr*(X2(K,1)^2-...                        
                   2*corr*X2(K,1)*X2(K,2)+X2(K,2)^2)-2*corr*(1-corr^2))/(1-corr^2)^2;               
               
            punt1=X2(K,1);
            punt2=X2(K,2);
            XX=[punt1 punt2];
            corrcorr=corr;
            NN=0;
            pivalue=3.1415926535897932;
            if punt1<=0 && punt2<=0 && corrcorr<=0                          
                aprime=punt1/(2*(1-corrcorr^2))^0.5;                        
                bprime=punt2/(2*(1-corrcorr^2))^0.5;               
                for I=1:4
                    for J=1:4
                        f=exp(aprime*(2*BB(I)-aprime)+bprime*(2*BB(J)-bprime)+2*corrcorr*(BB(I)-aprime)*(BB(J)-bprime));
                        NN=NN+AA(I)*AA(J)*f;       
                    end
                end
                CDFCDF=NN*(1-corrcorr^2)^0.5/pivalue;
            elseif punt1*punt2*corrcorr<=0         
                XX2=XX;
                corrcorr1=corrcorr;
                if punt1>=0
                    XX2(1,1)=-punt1;
                end
                if punt2>=0
                    XX2(1,2)=-punt2;
                end
                if corrcorr>=0
                    corrcorr1=-corrcorr;
                end               
                    aprime=XX2(1,1)/(2*(1-corrcorr1^2))^0.5;
                    bprime=XX2(1,2)/(2*(1-corrcorr1^2))^0.5;                
                for I=1:4
                    for J=1:4
                        f=exp(aprime*(2*BB(I)-aprime)+bprime*(2*BB(J)-bprime)+2*corrcorr1*(BB(I)-aprime)*(BB(J)-bprime));
                        NN=NN+AA(I)*AA(J)*f; 
                    end
                end   
                NN=NN*(1-corrcorr^2)^0.5/pivalue;
                if punt1<=0 && punt2>=0 && corrcorr>=0
                    x=-punt1;                                               
                    k=1/(1+0.2316419*x);                                    
                    Nprime=(1/(2*pivalue)^0.5)*exp(-x^2/2);
                    NCDF=Nprime*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5);                     
                    CDFCDF=NCDF-NN;
                elseif punt1>=0 && punt2<=0 && corrcorr>=0
                    x=-punt2;
                    k=1/(1+0.2316419*x);
                    Nprime=(1/(2*pivalue)^0.5)*exp(-x^2/2);
                    NCDF=Nprime*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5);                     
                    CDFCDF=NCDF-NN;                   
                elseif punt1>=0 && punt2>=0 && corrcorr<=0
                    x1=punt1;
                    k=1/(1+0.2316419*x1);
                    Nprime1=(1/(2*pivalue)^0.5)*exp(-x1^2/2);
                    NCDF1=1-Nprime1*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5); 
                    x2=punt2;
                    k=1/(1+0.2316419*x2);
                    Nprime2=(1/(2*pivalue)^0.5)*exp(-x2^2/2);
                    NCDF2=1-Nprime2*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5);                    
                    CDFCDF=NCDF1+NCDF2-1+NN;               
                end
            elseif punt1*punt2*corrcorr>0
                XX2=XX;
                CDFCDF=0;
                if punt1>=0
                    sgn1=1;
                else
                    sgn1=-1;
                end
                if punt2>=0
                    sgn2=1;
                else
                    sgn2=-1;
                end
                denum=(punt1^2-2*corrcorr*punt1*punt2+punt2^2)^0.5;                
                corrcorr1=((corrcorr*punt1-punt2)*sgn1)/denum;
                corrcorr2=((corrcorr*punt2-punt1)*sgn2)/denum;
                if corrcorr1>0.999999 || corrcorr1<-0.999999
                    if corrcorr1>0.999999
                        corrcorr1=0.999999;
                    else
                        corrcorr1=-0.999999;
                    end
                end
                if corrcorr2>0.999999 || corrcorr2<-0.999999
                    if corrcorr2>0.999999
                        corrcorr2=0.999999;
                    else
                        corrcorr2=-0.999999;
                    end
                end
                delta=(1-sgn1*sgn2)/4;
                for III=1:2
                    NN=0;
                    if III==1
                        XX2(1,2)=0;
                        corrcorr3=corrcorr1;
                    else
                        XX2(1,1)=0;
                        XX2(1,2)=punt2;
                        corrcorr3=corrcorr2; 
                    end                     
                    if XX2(1,1)<=0 && XX2(1,2)<=0 && corrcorr3<=0
                        aprime=XX2(1,1)/(2*(1-corrcorr3^2))^0.5;
                        bprime=XX2(1,2)/(2*(1-corrcorr3^2))^0.5;               
                        for I=1:4
                            for J=1:4
                                f=exp(aprime*(2*BB(I)-aprime)+bprime*(2*BB(J)-bprime)+2*corrcorr3*(BB(I)-aprime)*(BB(J)-bprime));
                                NN=NN+AA(I)*AA(J)*f;       
                            end
                        end
                        CDFF=NN*(1-corrcorr3^2)^0.5/pivalue;
                    else                    
                        corrcorr4=corrcorr3;
                        XX3=XX2;
                        if XX2(1,1)>=0
                            XX3(1,1)=-XX2(1,1);
                        end
                        if XX2(1,2)>=0
                            XX3(1,2)=-XX2(1,2);
                        end
                        if corrcorr3>=0
                            corrcorr4=-corrcorr3;
                        end 
                        aprime=XX3(1,1)/(2*(1-corrcorr4^2))^0.5;
                        bprime=XX3(1,2)/(2*(1-corrcorr4^2))^0.5;               
                        for I=1:4
                            for J=1:4
                                f=exp(aprime*(2*BB(I)-aprime)+bprime*(2*BB(J)-bprime)+2*corrcorr4*(BB(I)-aprime)*(BB(J)-bprime));
                                NN=NN+AA(I)*AA(J)*f; 
                            end
                        end   
                        NN=NN*(1-corrcorr4^2)^0.5/pivalue;
                        if XX2(1,1)<=0 && XX2(1,2)>=0 && corrcorr3>=0
                            x=-XX2(1,1);
                            k=1/(1+0.2316419*x);
                            Nprime=(1/(2*pivalue)^0.5)*exp(-x^2/2);
                            NCDF=Nprime*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5);                     
                            CDFF=NCDF-NN;                              
                        elseif XX2(1,1)>=0 && XX2(1,2)<=0 && corrcorr3>=0
                            x=-XX2(1,2);
                            k=1/(1+0.2316419*x);
                            Nprime=(1/(2*pivalue)^0.5)*exp(-x^2/2);
                            NCDF=Nprime*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5);                     
                            CDFF=NCDF-NN;       
                        elseif XX2(1,1)>=0 && XX2(1,2)>=0 && corrcorr3<=0
                            x1=XX2(1,1);
                            k=1/(1+0.2316419*x1);
                            Nprime1=(1/(2*pivalue)^0.5)*exp(-x1^2/2);
                            NCDF1=1-Nprime1*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5); 
                            x2=XX2(1,2);
                            k=1/(1+0.2316419*x2);
                            Nprime2=(1/(2*pivalue)^0.5)*exp(-x2^2/2);
                            NCDF2=1-Nprime2*(a1*k+a2*k^2+a3*k^3+a4*k^4+a5*k^5);                    
                            CDFF=NCDF1+NCDF2-1+NN;                                                                    
                        end                             
                    end
                    CDFCDF=CDFCDF+CDFF;
                end  
                CDFCDF=CDFCDF-delta;                 
            end                                                                                            
            [X2(K,5)]=CDFCDF;           
        end
        if Iteracion2==1                                                    
            for K=1:combumb
                cambio=0; 
                I=1;
                while cambio==0
                    if X(K,1)==X2(I,1) && X(K,2)==X2(I,2)
                        PDF(K)=X2(I,3);
                        GUV(K)=X2(I,4);
                        CDF(K)=X2(I,5);
                        cambio=1; 
                        Vector(K,1)=I;                                      
                    else
                        I=I+1;
                    end
                end
            end
        else
            for K=1:combumb
                PDF(K)=X2(Vector(K,1),3);                                   
                GUV(K)=X2(Vector(K,1),4);
                CDF(K)=X2(Vector(K,1),5);
            end
        end
        Derivada1sum=0;                                                     
        Derivada2sum=0;                                                             
        K=1; 
        for I=1:puntmax1
            for J=1:puntmax2         
                dens=PDF(K)-PDF(K+1)-PDF(K+2)+PDF(K+3);
                dist=CDF(K)-CDF(K+1)-CDF(K+2)+CDF(K+3)+10^-10;              
                guv =GUV(K)-GUV(K+1)-GUV(K+2)+GUV(K+3);
                Derivada1=TCONT(I,J)*dens/dist;                                                                     
                Derivada1sum=Derivada1sum+Derivada1;                           
                Derivada2=TCONT(I,J)*guv/dist-TCONT(I,J)*dens^2/dist^2;                               
                Derivada2sum=Derivada2sum+Derivada2;                        
                K=K+4;                
            end            
        end   
        if Iteracion<=NumIteraciones_N                                      
            if Derivada2sum==0
                Derivada2sum=0.0000000001;
            end
            corr2=corr-Derivada1sum/Derivada2sum;                          
            corrdif=abs(corr2-corr);            
            if corr2>=1                                                     
                L1=0.70;  L2=0.90;                                 
                corr2=(rand)*(L2-L1)+L1;  
                corrdif=1;
            elseif corr2<=-1
                L1=-0.70; L2=-0.90;                                 
                corr2=(rand)*(L2-L1)+L1; 
                corrdif=1;
            end
            Iteracion=Iteracion+1;    
        else                                                                
            if Iteracion1==0
                corrdif=1;
                Iteracion1=Iteracion1+1;
            else
                if Derivada1sum>0
                    corr2=(corr+corrsup)/2;
                    corrinf=corr;
                    Iteracion1=Iteracion1+1;
                else
                    corr2=(corr+corrinf)/2;
                    corrsup=corr;
                    Iteracion1=Iteracion1+1;
                end                 
            end
        end
        corr=corr2;
        CORR(1,2)=corr; CORR(2,1)=corr;  
        if Iteracion1==NumIteraciones_B                                     
            Iteracion1=0;
            Iteracion=0;
            corr_final=corr;
        elseif Iteracion1==1
            corr=corr_final;
        end
        Iteracion2=Iteracion2+1;
    end       
    MCORRPOLY(II,JJ)=corr;
    MCORRPOLY(JJ,II)=corr; 
end