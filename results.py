# visualize the results
from utils.plots import plot_results
import sys
geshu = len(sys.argv)
if geshu < 2:
    print('参数不足')
    sys.exit()
else:
     csv = sys.argv[1]
plot_results(file=csv, dir='')
