function I_norm = NormalizeImage (I)

  minI = min(min(I));
  maxI = max(max(I));
  
  I_norm = (I - minI) / (maxI - minI);

end