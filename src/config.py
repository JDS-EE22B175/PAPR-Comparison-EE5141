"""
Global config parameters for the simulation.
Based on Prof. Giridhar's slides.
"""

# System params
N = 256          # Total FFT size
K = 4            # Number of users
P = N // K       # subcarriers per user

assert K * P == N, "K*P must equal N"

# Simulation params
NUM_ITER = 10000        # Monte Carlo runs
L_OS = 4             # Oversampling factor for accurate PAPR
QAM_ORDERS = [4, 16, 64]   # modulation orders
RANDOM_SEED = 42

# To store the different schemes to run
SCHEME_NAMES = []
SCHEME_FUNCS = []

