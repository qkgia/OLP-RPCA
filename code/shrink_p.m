function y = shrink_p(x, T, p)
epsilon = eps(x);
if sum(abs(T(:)))==0
   y = x;
else
   y = sign(x) .* max(abs(x) - p * T * (abs(x) + epsilon) .^ (p - 1), 0) .* (abs(x) > T);
end

