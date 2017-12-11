function y = shrink_p(x, T, p)
epsilon = eps(x);
if sum(abs(T(:)))==0
   y = x;
else
   y = sign(x) .* max(abs(x) - T, 0);
end

