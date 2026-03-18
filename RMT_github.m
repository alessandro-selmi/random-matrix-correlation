% Correlation Analysis of Random Financial Matrices
% Alessandro Selmi
% Course: Problem Solving
% February 2024
% 
% This script implements a factor model to introduce synthetic 
% correlations into Gaussian random matrices, simulating financial 
% market behavior.
%
% Note: Note: For visual results, plots, and key formulas 
% (e.g., Wigner Surmise), please refer to the 'Presentation.pdf' 
% file in the repository.
%
%
% Key Objectives:
% 1. Eigenvalue Distribution vs. Marcenko-Pastur Density:
%   1.a Generate multivariate Gaussian random matrices.
%   1.b Compute covariance matrices and their empirical eigenvalue distributions.
%   1.c Validate results against the theoretical Marcenko-Pastur law.
% 2. Eigenvector Analysis:
%     Investigate eigenvectors corresponding to "outlier" eigenvalues (those 
%     deviating from the bulk) to identify dominant market factors or noise.
% 3. Level Spacing Statistics:
%     Compare the distribution of eigenvalue spacings (distances) with the 
%     Wigner Surmise, testing the universality of Random Matrix Theory (RMT).
%
%
% [ITA]
%
% Studio di matrici di correlazione di matrici generate randomicamente
%
% Obiettivi:
% 1. Confronto tra distribuzione degli autovalori e densità di Marcenko-Pastur
%   1.a Generazione matrici con elementi da gaussiane multivariate
%   1.b Calcolo matrici di covarianza con autovalori
%   1.c Confronto con la Marcenko-Pastur
% 2. Analisi degli autovettori corrispondenti agli autovalori che deviano dalla MP per mostrarne il significato
% 3. Confronto tra distribuzione degli spacing (cioè le distanze) tra autovalori e Wigner surmise.


clear all
close all


Nf = 7; % flag: n. of factors to use, valori previsti: 0, 1, 2, 3, 7

N = 100; % Number of stocks
T = 400; % Length of time series
q = N/T; % Rectangularity ratio
mu = 0;  % media dei dati
sigma = 1; % Standard deviation of idiosyncratic part of time series
Nm = 100; % Number of factor models to generate
str_Nm = sprintf('%d', Nm);
str_Nf = sprintf('%d', Nf);


len_I = linspace(0,20,21); % array delle lunghezze dell'intervallo per varianza di numero

L = [];       % Empty array to collect all eigenvalues
V1 = [];      % Empty array to collect leading eigenvectors
V2 = [];      % Empty array to collect second eigenvectors
V3 = [];      % Empty array to collect third eigenvectors
V4 = [];      % Empty array to collect fourth eigenvectors
V7 = [];      % Empty array to collect seventh eigenvectors
S_nn = [];    % Empty array to collect nearest neighbor spacings
S_nnn = [];   % Empty array to collect next nearest neighbor spacings
Csi = [];     % Empty array to collect unfolded eigenvalues
var_len_I = zeros(1,length(len_I));   % Iniz. array orizz. per varianza in funzione di len_I (singolo modello)
VAR_len_I = zeros(Nm,length(len_I));  % Iniz. array vert. di array orizz. di varianze (Nm modelli)
MU_VAR_len_I = [];                    % Iniz. array per medie sugli Nm modelli di varianza di numero in f. di len_I


%% 1. confronto tra distribuzione degli autovalori e densità di Marcenko-Pastur

switch(Nf)


    case 0

        for i = 1:Nm
        
            % Forming time series as idiosyncratic part plus factors
            Y = sigma*randn(N,T);

            C = corr(Y');
            dC = size(C,2); % restituisce la seconda dimensione della matrice dCxdC

            [eig_vectors,eig_values] = eig(C); % Eigenvalues and eigenvectors of last matrix used in loop

            l = diag(eig_values);
            l = sort(l);    
            a = 4; % Parameter of Gaussian broadening

            % f_ret = 0.98; % Fraction of eigenvalues to retain
            % l = l(round(dC*(1-f_ret)/2):round(dC*(1+f_ret)/2)); % Storing only the central f-fraction of eigenvalues

            % Inizio sezione unfolding
            ar_medie = l(a+1:end-a); % Array with means of Gaussians
            ar_devst = (l(2*a+1:end) - l(1:end-2*a))/2; % Array with std. deviations of Gaussians
    
            % Invoking function that computes the cumulative of the density with
            % Gaussian broadening (N.B. it's normalised in such a way that is
            % saturates to N, not to 1 as a standard cumulative). This gives us the
            % unfolded eigenvalues
            csi = gaussian_mix_cdf(ar_medie,ar_medie,ar_devst); 
            % Sorting unfolded eigenvalues
            csi = sort(csi);

            % varianza di numero    
            nu_csi = zeros(length(csi),length(len_I));    
            for k = 1:length(csi)
                % sum(abs(csi-csi(k))<(0.5.*len_I))  % genera per ogni csi(k) un array in cui vengono contati quanti csi sono in I(len_I)
                nu_csi(k,:) = sum(abs(csi-csi(k))<(0.5.*len_I));  % popol. mtx    
                var_len_I = mean((nu_csi-len_I).^2);  % popol. array di varianze, media fatta su colonne 
                VAR_len_I(i,:) = var_len_I(1,:);  % popol. array di array varianze
            end

            % disp('----------') % testo separatore
    
        
            L = [L; l];
            V1 = [V1; eig_vectors(:,1)];   
            V2 = [V2; eig_vectors(:,2)];
            V3 = [V3; eig_vectors(:,3)];
            V4 = [V4; eig_vectors(:,4)];
            Csi = [Csi; csi];   % Storing unfolded eigenvalues
            S_nn = [S_nn; diff(csi)];   % Storing nn spacings as differences between sorted unfolded eigenvalues
            S_nnn = [S_nnn; (csi(3:end)-csi(1:end-2))];  % Storing nnn spacings as differences between sorted unfolded eigenvalues
            MU_VAR_len_I = mean(VAR_len_I);  % popol. array con var medie
    
        end



case 1

        eps1 = 0.20; % Standard deviation of first factor

        for i = 1:Nm
        
            % Forming time series as idiosyncratic part plus factors
            Y = sigma*randn(N,T) + eps1*ones(N,T).*randn(1,T);

            C = corr(Y');
            dC = size(C,2); % restituisce la seconda dimensione della matrice dCxdC

            [eig_vectors,eig_values] = eig(C); % Eigenvalues and eigenvectors of last matrix used in loop

            l = diag(eig_values);
            l = sort(l);    
            a = 4; % Parameter of Gaussian broadening

            % f_ret = 0.98; % Fraction of eigenvalues to retain
            % l = l(round(dC*(1-f_ret)/2):round(dC*(1+f_ret)/2)); % Storing only the central f-fraction of eigenvalues

            v1 = [eig_vectors(:,1)]; % Primo autovettore
            if(mean(v1)<0)
                v1 = -v1;   % Cambia segno se è a media negativa, vedi istogramma
            end

            % Inizio sezione unfolding
            ar_medie = l(a+1:end-a); % Array with means of Gaussians
            ar_devst = (l(2*a+1:end) - l(1:end-2*a))/2; % Array with std. deviations of Gaussians
    
            % Invoking function that computes the cumulative of the density with
            % Gaussian broadening (N.B. it's normalised in such a way that is
            % saturates to N, not to 1 as a standard cumulative). This gives us the
            % unfolded eigenvalues
            csi = gaussian_mix_cdf(ar_medie,ar_medie,ar_devst); 
            % Sorting unfolded eigenvalues
            csi = sort(csi);

            % varianza di numero    
            nu_csi = zeros(length(csi),length(len_I));    
            for k = 1:length(csi)
                % sum(abs(csi-csi(k))<(0.5.*len_I))  % genera per ogni csi(k) un array in cui vengono contati quanti csi sono in I(len_I)
                nu_csi(k,:) = sum(abs(csi-csi(k))<(0.5.*len_I));  % popol. mtx    
                var_len_I = mean((nu_csi-len_I).^2);  % popol. array di varianze, media fatta su colonne 
                VAR_len_I(i,:) = var_len_I(1,:);  % popol. array di array varianze
            end

            % disp('----------') % testo separatore
    
        
            L = [L; l];
            V1 = [V1; v1];   
            V2 = [V2; eig_vectors(:,2)];
            V3 = [V3; eig_vectors(:,3)];
            V4 = [V4; eig_vectors(:,4)];
            Csi = [Csi; csi];   % Storing unfolded eigenvalues
            S_nn = [S_nn; diff(csi)];   % Storing nn spacings as differences between sorted unfolded eigenvalues
            S_nnn = [S_nnn; (csi(3:end)-csi(1:end-2))];  % Storing nnn spacings as differences between sorted unfolded eigenvalues
            MU_VAR_len_I = mean(VAR_len_I);  % popol. array con var medie
    
        end



    case 2

        eps1 = 0.20; % Standard deviation of first factor
        eps2 = 0.25; % Standard deviation of second factor
        frac1 = 0.60; % Fraction of variables affected by first factor

        for i = 1:Nm
 
            aux1 = ones(N,T).*randn(1,T); % First factor
            aux1(1:round(frac1*N),:) = 0; % Selecting first f*N variables as those affected by first factor
            aux2 = ones(N,T).*randn(1,T); % Second factor
            aux2(round(frac1*N)+1:end,:) = 0; % Selecting remaining variables as those affected by second factor 
        
            % Forming time series as idiosyncratic part plus factors
            Y = sigma*randn(N,T) + eps1*aux1 + eps2*aux2;

            C = corr(Y');
            dC = size(C,2); % restituisce la seconda dimensione della matrice dCxdC

            [eig_vectors,eig_values] = eig(C); % Eigenvalues and eigenvectors of last matrix used in loop

            l = diag(eig_values);
            l = sort(l);    
            a = 4; % Parameter of Gaussian broadening

            % f_ret = 0.98; % Fraction of eigenvalues to retain
            % l = l(round(dC*(1-f_ret)/2):round(dC*(1+f_ret)/2)); % Storing only the central f-fraction of eigenvalues

            v1 = [eig_vectors(:,1)]; % Primo autovettore
            if(mean(v1)<0)
                v1 = -v1;   % Cambia segno se è a media negativa, vedi istogramma
            end

            v2 = [eig_vectors(:,2)]; % Secondo autovettore
            if(mean(v2)<0)
                v2 = -v2;   % Cambia segno se è a media negativa, vedi istogramma
            end


            % Inizio sezione unfolding
            ar_medie = l(a+1:end-a); % Array with means of Gaussians
            ar_devst = (l(2*a+1:end) - l(1:end-2*a))/2; % Array with std. deviations of Gaussians
    
            % Invoking function that computes the cumulative of the density with
            % Gaussian broadening (N.B. it's normalised in such a way that is
            % saturates to N, not to 1 as a standard cumulative). This gives us the
            % unfolded eigenvalues
            csi = gaussian_mix_cdf(ar_medie,ar_medie,ar_devst); 
            % Sorting unfolded eigenvalues
            csi = sort(csi);

            % varianza di numero    
            nu_csi = zeros(length(csi),length(len_I));    
            for k = 1:length(csi)
                % sum(abs(csi-csi(k))<(0.5.*len_I))  % genera per ogni csi(k) un array in cui vengono contati quanti csi sono in I(len_I)
                nu_csi(k,:) = sum(abs(csi-csi(k))<(0.5.*len_I));  % popol. mtx    
                var_len_I = mean((nu_csi-len_I).^2);  % popol. array di varianze, media fatta su colonne 
                VAR_len_I(i,:) = var_len_I(1,:);  % popol. array di array varianze
            end

            % disp('----------') % testo separatore
    
        
            L = [L; l];
            V1 = [V1; v1];   
            V2 = [V2; v2];   
            V3 = [V3; eig_vectors(:,3)];
            V4 = [V4; eig_vectors(:,4)];
            Csi = [Csi; csi];   % Storing unfolded eigenvalues
            S_nn = [S_nn; diff(csi)];   % Storing nn spacings as differences between sorted unfolded eigenvalues
            S_nnn = [S_nnn; (csi(3:end)-csi(1:end-2))];  % Storing nnn spacings as differences between sorted unfolded eigenvalues
            MU_VAR_len_I = mean(VAR_len_I);  % popol. array con var medie

        end



    case 3

        eps1 = 0.22; % Standard deviation of first factor
        eps2 = 0.30; % Standard deviation of second factor
        eps3 = 0.35; % Standard deviation of third factor
        frac1 = 0.40; % Fraction of variables affected by first factor
        frac2 = 0.30; % Fraction of variables affected by second factor (and not first)

        for i = 1:Nm
 
            aux1 = ones(N,T).*randn(1,T);                           % First factor
            aux1(1:round(frac1*N),:) = 0;                           % Selecting first f*N variables as those affected by first factor
            aux2 = ones(N,T).*randn(1,T);                           % Second factor
            aux2(round(frac1*N)+1:round((frac1+frac2)*N)+1,:) = 0;  % Selecting variables affected by second factor
            aux3 = ones(N,T).*randn(1,T);                           % Third factor
            aux3(round((frac1+frac2)*N)+2:end,:) = 0;               % Selecting remaining variables as those affected by third factor
        
            % Forming time series as idiosyncratic part plus factors
            Y = sigma*randn(N,T) + eps1*aux1 + eps2*aux2 + eps3*aux3;

            C = corr(Y');
            dC = size(C,2); % restituisce la seconda dimensione della matrice dCxdC

            [eig_vectors,eig_values] = eig(C); % Eigenvalues and eigenvectors of last matrix used in loop

            l = diag(eig_values);
            l = sort(l);    
            a = 4; % Parameter of Gaussian broadening

            % f_ret = 0.98; % Fraction of eigenvalues to retain
            % l = l(round(dC*(1-f_ret)/2):round(dC*(1+f_ret)/2)); % Storing only the central f-fraction of eigenvalues

            v1 = [eig_vectors(:,1)]; % Primo autovettore
            if(mean(v1)<0)
                v1 = -v1;   % Cambia segno se è a media negativa, vedi istogramma
            end

            v2 = [eig_vectors(:,2)]; % Secondo autovettore
            if(mean(v2)<0)
                v2 = -v2;   % Cambia segno se è a media negativa, vedi istogramma
            end

            v3 = [eig_vectors(:,3)]; % Secondo autovettore
            if(mean(v3)<0)
                v3 = -v3;   % Cambia segno se è a media negativa, vedi istogramma
            end


            % Inizio sezione unfolding
            ar_medie = l(a+1:end-a); % Array with means of Gaussians
            ar_devst = (l(2*a+1:end) - l(1:end-2*a))/2; % Array with std. deviations of Gaussians
    
            % Invoking function that computes the cumulative of the density with
            % Gaussian broadening (N.B. it's normalised in such a way that is
            % saturates to N, not to 1 as a standard cumulative). This gives us the
            % unfolded eigenvalues
            csi = gaussian_mix_cdf(ar_medie,ar_medie,ar_devst); 
            % Sorting unfolded eigenvalues
            csi = sort(csi);

            % varianza di numero    
            nu_csi = zeros(length(csi),length(len_I));    
            for k = 1:length(csi)
                % sum(abs(csi-csi(k))<(0.5.*len_I))  % genera per ogni csi(k) un array in cui vengono contati quanti csi sono in I(len_I)
                nu_csi(k,:) = sum(abs(csi-csi(k))<(0.5.*len_I));  % popol. mtx    
                var_len_I = mean((nu_csi-len_I).^2);  % popol. array di varianze, media fatta su colonne 
                VAR_len_I(i,:) = var_len_I(1,:);  % popol. array di array varianze
            end

            % disp('----------') % testo separatore
    
        
            L = [L; l];
            V1 = [V1; v1];   
            V2 = [V2; v2];
            V3 = [V3; v3];
            V4 = [V4; eig_vectors(:,4)];
            Csi = [Csi; csi];   % Storing unfolded eigenvalues
            S_nn = [S_nn; diff(csi)];   % Storing nn spacings as differences between sorted unfolded eigenvalues
            S_nnn = [S_nnn; (csi(3:end)-csi(1:end-2))];  % Storing nnn spacings as differences between sorted unfolded eigenvalues
            MU_VAR_len_I = mean(VAR_len_I);  % popol. array con var medie

        end


        
    case 7

        eps1 = 0.05; % Standard deviation of first factor
        eps2 = 1.21; % Standard deviation of second factor
        eps3 = 2.17; % Standard deviation of third factor
        eps4 = 0.13; % Standard deviation of fourth factor
        eps5 = 1.09; % Standard deviation of fifth factor
        eps6 = 2.01; % Standard deviation of sixth factor
        eps7 = 1.55; % Standard deviation of seventh factor
        frac1 = 0.18; % Fraction of variables affected by first factor
        frac2 = 0.15; % Fraction of variables affected by second factor (and not first)
        frac3 = 0.11; % Fraction of variables affected by third factor (and not first or second)
        frac4 = 0.12; % Fraction of variables affected by fourth factor (only)
        frac5 = 0.11; % Fraction of variables affected by fifth factor (only)
        frac6 = 0.15; % Fraction of variables affected by sixth factor (only)

        for i = 1:Nm
 
            aux1 = ones(N,T).*randn(1,T);                           % First factor
            aux1(1:round(frac1*N),:) = 0;                           % Selecting first f*N variables as those affected by first factor
            n_fin2 = round((frac1+frac2)*N)+1;
            aux2 = ones(N,T).*randn(1,T);                           % Second factor
            aux2(round(frac1*N)+1:n_fin2,:) = 0;                    % Selecting variables affected by second factor
            n_fin3 = round((frac1+frac2+frac3)*N)+2;
            aux3 = ones(N,T).*randn(1,T);                           % Third factor
            aux3(n_fin2+1:n_fin3,:) = 0;
            n_fin4 = round((frac1+frac2+frac3+frac4)*N)+3;
            aux4 = ones(N,T).*randn(1,T);                           
            aux4(n_fin3+1:n_fin4,:) = 0;   
            n_fin5 = round((frac1+frac2+frac3+frac4+frac5)*N)+4;
            aux5 = ones(N,T).*randn(1,T);                           
            aux5(n_fin4+1:n_fin5,:) = 0;  
            n_fin6 = round((frac1+frac2+frac3+frac4+frac5+frac6)*N)+5;
            aux6 = ones(N,T).*randn(1,T);                           
            aux6(n_fin5+1:n_fin6,:) = 0;
            aux7 = ones(N,T).*randn(1,T);                           
            aux7(n_fin6+1:end,:) = 0;               
        
            % Forming time series as idiosyncratic part plus factors
            Y = sigma*randn(N,T) + eps1*aux1 + eps2*aux2 + eps3*aux3 + eps4*aux4 + eps5*aux5 + eps6*aux6 + eps7*aux7;

            C = corr(Y');
            dC = size(C,2); % restituisce la seconda dimensione della matrice dCxdC

            [eig_vectors,eig_values] = eig(C); % Eigenvalues and eigenvectors of last matrix used in loop

            l = diag(eig_values);
            l = sort(l);    
            a = 4; % Parameter of Gaussian broadening

            % f_ret = 0.98; % Fraction of eigenvalues to retain
            % l = l(round(dC*(1-f_ret)/2):round(dC*(1+f_ret)/2)); % Storing only the central f-fraction of eigenvalues

            v1 = [eig_vectors(:,1)]; % Primo autovettore
            if(mean(v1)<0)
                v1 = -v1;   % Cambia segno se è a media negativa, vedi istogramma
            end

            v2 = [eig_vectors(:,2)]; % Secondo autovettore
            if(mean(v2)<0)
                v2 = -v2;   % Cambia segno se è a media negativa, vedi istogramma
            end

            v3 = [eig_vectors(:,3)]; % Secondo autovettore
            if(mean(v3)<0)
                v3 = -v3;   % Cambia segno se è a media negativa, vedi istogramma
            end

            v4 = [eig_vectors(:,4)]; % Secondo autovettore
            if(mean(v4)<0)
                v4 = -v4;   % Cambia segno se è a media negativa, vedi istogramma
            end

            v7 = [eig_vectors(:,7)]; % Secondo autovettore
            if(mean(v7)<0)
                v7 = -v7;   % Cambia segno se è a media negativa, vedi istogramma
            end


            % Inizio sezione unfolding
            ar_medie = l(a+1:end-a); % Array with means of Gaussians
            ar_devst = (l(2*a+1:end) - l(1:end-2*a))/2; % Array with std. deviations of Gaussians
    
            % Invoking function that computes the cumulative of the density with
            % Gaussian broadening (N.B. it's normalised in such a way that is
            % saturates to N, not to 1 as a standard cumulative). This gives us the
            % unfolded eigenvalues
            csi = gaussian_mix_cdf(ar_medie,ar_medie,ar_devst); 
            % Sorting unfolded eigenvalues
            csi = sort(csi);

            % varianza di numero    
            nu_csi = zeros(length(csi),length(len_I));    
            for k = 1:length(csi)
                % sum(abs(csi-csi(k))<(0.5.*len_I))  % genera per ogni csi(k) un array in cui vengono contati quanti csi sono in I(len_I)
                nu_csi(k,:) = sum(abs(csi-csi(k))<(0.5.*len_I));  % popol. mtx    
                var_len_I = mean((nu_csi-len_I).^2);  % popol. array di varianze, media fatta su colonne 
                VAR_len_I(i,:) = var_len_I(1,:);  % popol. array di array varianze
            end
            

            % disp('----------') % testo separatore
    
        
            L = [L; l];
            V1 = [V1; v1];   
            V2 = [V2; v2];
            V3 = [V3; v3];
            V4 = [V4; v4];
            V7 = [V7; v7];
            Csi = [Csi; csi];   % Storing unfolded eigenvalues
            S_nn = [S_nn; diff(csi)];   % Storing nn spacings as differences between sorted unfolded eigenvalues
            S_nnn = [S_nnn; (csi(3:end)-csi(1:end-2))];  % Storing nnn spacings as differences between sorted unfolded eigenvalues
            MU_VAR_len_I = mean(VAR_len_I);  % popol. array con var medie
    

        end

    otherwise
        tim = datetime('now','Format','HH:mm:ss');
        disp('Numero di fattori inserito non previsto: abortendo alle ore')
        disp(tim)
        return

end


% Plotting histogram of all eigenvalues
    figure('Name','Distribuzione degli Autovalori', 'NumberTitle','off');
    histogram(L,'Normalization','pdf');
    %histo_aval = histogram(L,'Normalization','pdf');
    % histo_aval.NumBins = morebins(histo_aval);
    %histo_aval.NumBins = 250;
    %histo_aval

hold on

% Defining and plotting Marcenko-Pastur distribution
    lam_p = (1+sqrt(q))^2; %*sigma^2;
    lam_m = (1-sqrt(q))^2; %*sigma^2;
    x = linspace(lam_m,lam_p,1000);
    p = sqrt((x-lam_m).*(lam_p-x))./(2*pi*q*x); %sigma^2*x);
    plot(x,p,'b-','Linewidth',2)
    xlabel('$\lambda$','Interpreter','Latex')
    ylabel('$p(\lambda)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Distribuzione degli Autovalori')
    lgd_avl = legend('Frequenza osservata','Distribuzione di Marcenko-Pastur');
    lgd_avl.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];



%% 2. Analisi dei primi autovettori


% Defining Porter-Thomas law
    sPT = 1/sqrt(N); % Standard deviation of Porter-Thomas law
    x = linspace(-5*sPT,5*sPT,1000);
    pPT = exp(-x.^2/(2*sPT^2))/sqrt(2*pi*sPT^2);


% Plotting histogram of 1st eigenvectors
    figure('Name','Componenti del Primo Autovettore','NumberTitle','off');
    histogram(V1,'Normalization','pdf')
    xlabel('$x$','Interpreter','Latex')
    ylabel('$p^{(1)}_{avt}(x)$','Interpreter','Latex')
    title('Componenti del Primo Autovettore')
    lgd_avt1 = legend('Frequenza osservata');
    lgd_avt1.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];

%{
hold on
    
% Plotting Porter-Thomas law
    plot(x,pPT,'b-','Linewidth',2)
    xlabel('$x$','Interpreter','Latex')
    ylabel('$p^{(1)}_{avt}(x)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Componenti del Primo Autovettore')
    lgd_avt1 = legend('Frequenza osservata','Legge di Porter-Thomas');
    lgd_avt1.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];
%}


% Plotting histogram of 2nd eigenvectors
    figure('Name','Componenti del Secondo Autovettore','NumberTitle','off');
    histogram(V2,'Normalization','pdf')

hold on
    
% Plotting Porter-Thomas law
    plot(x,pPT,'b-','Linewidth',2)
    xlabel('$x$','Interpreter','Latex')
    ylabel('$p^{(2)}_{avt}(x)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Componenti del Secondo Autovettore')
    lgd_avt2 = legend('Frequenza osservata','Legge di Porter-Thomas');
    lgd_avt2.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];


% Plotting histogram of 3rd eigenvectors
    figure('Name','Componenti del Terzo Autovettore','NumberTitle','off');
    histogram(V3,'Normalization','pdf')

hold on
    
% Plotting Porter-Thomas law
    plot(x,pPT,'b-','Linewidth',2)
    xlabel('$x$','Interpreter','Latex')
    ylabel('$p^{(3)}_{avt}(x)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Componenti del Terzo Autovettore')
    lgd_avt3 = legend('Frequenza osservata','Legge di Porter-Thomas');
    lgd_avt3.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];


% Plotting histogram of 4th eigenvectors
    figure('Name','Componenti del Quarto Autovettore','NumberTitle','off');
    histogram(V4,'Normalization','pdf')

hold on
    
% Plotting Porter-Thomas law
    plot(x,pPT,'b-','Linewidth',2)
    xlabel('$x$','Interpreter','Latex')
    ylabel('$p^{(4)}_{avt}(x)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Componenti del Quarto Autovettore')
    lgd_avt4 = legend('Frequenza osservata','Legge di Porter-Thomas');
    lgd_avt4.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];


if Nm==7
% Plotting histogram of 7th eigenvectors
    figure('Name','Componenti del Settimo Autovettore','NumberTitle','off');
    histogram(V7,'Normalization','pdf')

hold on
    
% Defining and plotting Porter-Thomas law
    plot(x,pPT,'b-','Linewidth',2)
    xlabel('$x$','Interpreter','Latex')
    ylabel('$p^{(7)}_{avt}(x)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Componenti del Settimo Autovettore')
    lgd_avt7 = legend('Frequenza osservata','Legge di Porter-Thomas');
    lgd_avt7.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];
end



%% 3. Studio degli spacing


%{
% Plotting histogram of unfolded eigenvalues
figure('Name','Distribuz. Autovalori Unfoldati','NumberTitle','off');
histogram(Csi,'Normalization','pdf')
%}


% Nearest neighbor

% Plot degli spacing
    figure('Name','Spacing Nearest Neighbor','NumberTitle','off');
    histogram(S_nn,'Normalization','pdf')
  
hold on

% Defining and plotting Wigner surmise
    sp1 = linspace(0,3,1000);
    pWS = 0.5*pi*sp1.*exp(-0.25*pi*sp1.^2);
    plot(sp1,pWS,'b-','Linewidth',2)
    xlabel('$spacing$','Interpreter','Latex')
    ylabel('$p_{nn}(s)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Spacing Nearest Neighbor')
    lgd_snn = legend('Frequenza osservata','Wigner surmise');
    lgd_snn.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];



% Next nearest neighbor

% Plot degli spacing
    figure('Name','Spacing Next Nearest Neighbor','NumberTitle','off');
    histogram(0.5*S_nnn,'Normalization','pdf') % *0.5 per considerare uno spacing medio
  
hold on

% Defining and plotting GSE curve
    sp2 = linspace(0,3,1000);
    pGSE = (2^18)*((9*pi)^(-3))*(sp2.^4).*exp(-64*((9*pi)^(-1))*(sp2.^2));
    plot(sp2,pGSE,'b-','Linewidth',2)
    xlabel('$spacing$','Interpreter','Latex')
    ylabel('$p_{nnn}(s)$','Interpreter','Latex')
    set(gca,'FontSize',20)
    title('Spacing Next Nearest Neighbor')
    lgd_snnn = legend('Frequenza osservata','Distribuzione attesa');
    lgd_snnn.Title.String = ['Simulazione a ' str_Nm ' modelli a ' str_Nf ' fattori'];




%{
% Varianza di numero
    figure('Name','Varianza di numero','NumberTitle','off');
    plot(len_I,MU_VAR_len_I,'b .','Linewidth',2)
    xlabel('$len_I$','Interpreter','Latex')
    ylabel('$\Sigma^2(len_I)$','Interpreter','Latex')
    set(gca,'FontSize',20)

    MU_VAR_su_lnlen = MU_VAR_len_I./(log(len_I));
    figure('Name','Var di numero fratto log naturale di lunghezza di I','NumberTitle','off');
    plot(len_I,MU_VAR_su_lnlen,'b .','Linewidth',2)
    xlabel('$len_I$','Interpreter','Latex')
    ylabel('$\frac{\Sigma^2(len_I)}{ln(len_I)}$','Interpreter','Latex')
    set(gca,'FontSize',20)
%}