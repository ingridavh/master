##########################################
# A benchmark tester of Python time      #
##########################################

import numpy as np
import time

print "Hello World! I'm the time tester."
t0 = time.clock()
print t0

t1 = time.time()
for i in range(1000):
    print "Just killing time"

print time.time() - t1
