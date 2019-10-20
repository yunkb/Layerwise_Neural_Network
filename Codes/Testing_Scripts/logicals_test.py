import numpy as np

a_1 = np.arange(10)
a_2 = np.linspace(11,20,10)
b = np.vstack([a_1.T,a_2.T])
b = b.T
test_x = np.where(np.logical_or(b[:,0] == 0,b[:,0] == 4), np.ones(b.shape[0]), 2*np.ones(b.shape[0]))
test_y = np.where(np.logical_or(b[:,1] == 13,b[:,1] == 16), np.ones(b.shape[0]), 2*np.ones(b.shape[0]))

test_x2 = np.where(np.logical_or(b == 0,b == 4), np.ones((b.shape[0], b.shape[1])), 0*np.ones((b.shape[0], b.shape[1])))
test_y2 = np.where(np.logical_or(b == 13,b == 16), np.ones((b.shape[0], b.shape[1])), 0*np.ones((b.shape[0], b.shape[1])))

c1 = b*test_x2
c2 = b*test_y2