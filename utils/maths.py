import numpy as np

def _count(X):
    try:
        X = np.astype('float')
        X = X[~np.isnan(X)]
        return len(X)
    except :
        return len(X)
    
def _mean(X):
    add = 0
    for x in X:
        if np.isnan(x):
            continue
        add += x
    return add/len(X)

def _std(X):
  mean = _mean(X)
  total = 0
  for x in X:
    if np.isnan(x):
      continue
    total = total + (x - mean) ** 2
  return (total / len(X)) ** 0.5

def _min(X):
  min_value = X[0]
  for x in X:
    val = x
    if val < min_value:
      min_value = val
  return min_value

def _max(X):
  min_value = X[0]
  for x in X:
    val = x
    if val > min_value:
      min_value = val
  return min_value

def _percentile(X, p):
  X.sort()
  k = (len(X) - 1) * (p / 100)
  f = np.floor(k)
  c = np.ceil(k)

  if f == c:
    return X[int(k)]

  d0 = X[int(f)] * (c - k)
  d1 = X[int(c)] * (k - f)
  return d0 + d1
