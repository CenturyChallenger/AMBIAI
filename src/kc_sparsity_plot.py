## Goal: This replaces kc_sparsity_plot.py. It simulates the MB training process to show how ifn_thresh (Sparsity) and INPUT_SCALE affect the model's ability to "remember" a route.

import numpy as np
import matplotlib.pyplot as plt

class MBSimulator:
    def __init__(self, num_vpns=320, num_kcs=20000, num_pre=10):
        self.num_vpns = num_vpns
        self.num_kcs = num_kcs
        # Sparse random connectivity: VPN -> KC
        self.weights_vpn_kc = np.random.choice([0, 1], size=(num_kcs, num_vpns), p=[1-(num_pre/num_vpns), num_pre/num_vpns])

    def get_kc_activity(self, input_pattern, ifn_thresh, input_scale):
        # 1. Drive from VPNs
        drive = np.dot(self.weights_vpn_kc, input_pattern * input_scale)
        # 2. Approximation of the inhibitory feedback (ifn_thresh)
        # In the workshop, ifn_thresh is the threshold for inhibition. 
        # Lower threshold = more inhibition = higher sparsity.
        # We simulate this by taking the top K neurons where K is modulated by ifn_thresh.
        k = int(max(1, (1000 / ifn_thresh) * 100)) # Mapping threshold to # of active cells
        active_indices = np.argsort(drive)[-k:]
        kc_spikes = np.zeros(self.num_kcs)
        kc_spikes[active_indices] = 1
        return kc_spikes

    def run_sweep(self, scales, thresholds):
        results = np.zeros((len(scales), len(thresholds)))
        # Generate dummy "route" and "novel" images
        route_img = np.random.rand(self.num_vpns)
        novel_img = np.random.rand(self.num_vpns)
        
        for i, scale in enumerate(scales):
            for j, thresh in enumerate(thresholds):
                # Train: Anti-Hebbian learning on the route image
                kc_active = self.get_kc_activity(route_img, thresh, scale)
                w_mbon = np.ones(self.num_kcs)
                w_mbon[kc_active > 0] -= 0.1 # Plasticity: weaken active synapses
                
                # Test: Compare familiar vs novel
                fam_resp = np.sum(self.get_kc_activity(route_img, thresh, scale) * w_mbon)
                nov_resp = np.sum(self.get_kc_activity(novel_img, thresh, scale) * w_mbon)
                
                # Accuracy metric: Discrimination Index
                results[i, j] = (nov_resp - fam_resp) / (nov_resp + 1e-9)
        return results

# Integration Code for Notebook
def plot_sparsity_sweep():
    sim = MBSimulator()
    scales = [0.001, 0.005, 0.01]
    thresholds = np.linspace(50, 500, 10) # Sweep ifn_thresh
    
    data = sim.run_sweep(scales, thresholds)
    
    plt.figure(figsize=(10, 5))
    for i, scale in enumerate(scales):
        plt.plot(thresholds, data[i, :], label=f'Input Scale: {scale}', marker='o')
    
    plt.xlabel("Inhibitory Threshold (ifn_thresh)")
    plt.ylabel("Navigation Accuracy (Discrimination)")
    plt.title("Effect of KC Sparsity & Input Scale on Familiarity Detection")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run this in a notebook cell
plot_sparsity_sweep()
