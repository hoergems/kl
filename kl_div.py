import numpy as np

class KL:
    def __init__(self):
        cov0 = np.array([[0.1, 0.0, 0.0],
                         [0.0, 0.1, 0.0],
                         [0.0, 0.0, 0.1]])
        cov1 = np.array([[0.2, 0.0, 0.0],
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.2]])
        mean0 = np.array([0.1, 0.0, 0.0])
        mean1 = np.array([0.0, 0.2, 0.0])
        
        k1 = self.calc_kl(mean0, mean1, cov0, cov1)
        
        mean0_t, cov0_t = self.propagate(mean0, cov0)        
        mean1_t, cov1_t = self.propagate(mean1, cov1)
        k2 = self.calc_kl(mean0_t, mean1_t, cov0_t, cov1_t)
        print "det " + str(np.linalg.det(self.A()))
        print k1
        print k2
               
    
    def calc_kl(self, mean0, mean1, cov0, cov1):
        cov1_inv = np.linalg.inv(cov1)
        kl = ((1.0 / 2.0) * 
              (np.trace(np.dot(cov1_inv, cov0)) + 
               np.dot(np.transpose(mean1 - mean0), np.dot(cov1_inv, mean1 - mean0)) - 3.0 +
               np.log(np.linalg.det(cov1) / np.linalg.det(cov0))))
        return kl
    
    def propagate(self, mean, cov):
        A = self.A()
        A_inv = np.linalg.inv(A)
        mean_new = np.dot(A, mean)
        cov_new = np.dot(A, np.dot(cov, A_inv))
        return mean_new, cov_new
    
    def A(self):
        A = np.array([[1.0, 2.0, 0.0],
                      [0.1, 0.2, 0.3],
                      [0.0, 1.0, 1.0]])
        return A

if __name__ == "__main__":
    KL()