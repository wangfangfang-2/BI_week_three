from scipy import stats
mu = 179.5
sigma = 3.697
x = 180
prob = stats.norm.pdf(x, mu, sigma)
print(prob)

import numpy as np
print(np.log2(0.4))
print(np.log2(0.2))
#以下为结果
zhangyuxi@ZhangdeMacBook-Pro L3 % python3 NORMDIST.py 
0.10692733469896672
-1.3219280948873622
-2.321928094887362