%Homework 2, Matlab computer exercise
%Description: This program trains a single delta rule neuron using the
%input matrix x, and the desired output d, with initial values of w
%and b, and a learning rate of a. !
%Initialize variables
x = [2,1;1,-2;-1,1]; %Input vectors
d = [1,-1]; %Desired output
w = [.4 -.1 .3]; %Initial weights
b = .2; %Initial bias
a = .2; %Learning rate !
%Sweep through the network twice
n = 100;
f_x = zeros(1,n);
last = 0;
tol = 0.0001;
stopped_at = n;
for i = 1:n
%Use each input-output pair to update the weights and bias
for j = 1:2
%Calculate the output
y = tanh(w*x(:,j)+b);
%Find the generalized error
z = (d(j) - y)*(1-tanh(y)^2);
%Calculate the change in weights and bias
dw = a*z*x(:,j)';
db = -a*z;
%Update the weights and bias
w = w + dw;
b = b + db;
end
y = tanh(w*x+b);
curr = norm(y-d,2)^2;
if abs(last-curr) < tol
    abs(last-curr)
    stopped_at = i
    break
end
last = curr;
f_x(i) = norm(y-d,2)^2;
end 
%Display the results
dw
db
w
b 
y = tanh(w*x+b)
x = linspace(1,stopped_at,stopped_at);
plot(x,f_x(1:stopped_at));