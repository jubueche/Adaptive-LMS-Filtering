clear all
close all
hold off
                                                    %channel system order
sysorder = 5 ;
                                                    %Number of system points
N=2000;
inp = randn(N,1);
n = randn(N,1);
[b,a] = butter(2,0.25);                             %Create a low pass filter of second order with cutoff frequency 0.25
Gz = tf(b,a,.1);                                    %Create discrete-time transfer function with undetermined sample time
                                                    %and numerator b and a, which are the transfer function coefficients of the
                                                    %2nd order butterworth filter with cutoff freq 0.25
                                                    % if you use ldiv this will give h :filter weights to be
h=  [0.0976;                                        %This is the actual filter that we are trying to recreate
    0.2873;
    0.3360;
    0.2210;
    0.0964;];
y = lsim(Gz,inp);                                   %This simulates the time response of the system Gz given random input inp
                                                    %add some noise
n = n * std(y)/(10*std(n));
d = y + n;
totallength=size(d,1);
                                                    %Take 60 points for training
N=60 ;
                                                    %begin of algorithm
w = zeros ( sysorder  , 1 ) ;
for n = sysorder : N 
	u = inp(n:-1:n-sysorder+1) ;
    y(n)= w' * u;                                   %Compute y(n) with the weights and the input
    e(n) = d(n) - y(n) ;                            %Compute the error: d(n) is the true input
                                                    %Start with big mu for speeding the convergence then slow down
                                                    %to reach the correct weights
    if n < 20
        mu=0.32;
    else
        mu=0.15;
    end
                                                    %The update corresponds to the LMS update rule, which can be derived
                                                    %using the gradient of the cost function and the approximation of the
                                                    %expectation function E{e(n)x(n)} which is approximated by the average
                                                    %over n points. In the lms we use n=1, so just e(n)x(n)
	w = w + mu * u * e(n) ;                         %Update the weights w.r.t. the old weight,learning rate mu, input u, and error e(n)
end 
                                                    %Check of results on
                                                    %the rest of the data
for n =  N+1 : totallength
	u = inp(n:-1:n-sysorder+1);
    y(n) = w' * u ;                                 %Compute the results with the obtained weight vector
    e(n) = d(n) - y(n) ;                            %Compute the error
end 
hold on
plot(d)
plot(y,'r');                                        %Plot the obtained output vs. the true output d
title('System output') ;
xlabel('Samples')
ylabel('True and estimated output')
figure
semilogy((abs(e))) ;                                %Create semi log scale plot (only y axis)
title('Error curve') ;
xlabel('Samples')
ylabel('Error value')
figure
                                                    %Plot the weights h and w and let the user see the difference
plot(h, 'k+')
hold on
plot(w, 'r*')
legend('Actual weights','Estimated weights')
title('Comparison of the actual weights and the estimated weights') ;
axis([0 6 0.05 0.35])