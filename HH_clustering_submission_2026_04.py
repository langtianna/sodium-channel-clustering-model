#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Experiment 04: Channel Clustering on a Full Na/K Excitable Membrane Model

Goal:
Build a more realistic excitable-cell model using competing sodium, potassium,
and leak currents before testing whether sodium channel clustering changes
action potential reliability.

Why this notebook exists:
The earlier simplified clustering model entered a persistent depolarized state
and could not meaningfully distinguish spike success across cluster fractions.
This notebook rebuilds the experiment on a more complete Hodgkin-Huxley-style
framework.

Main objectives:
1. Confirm that the model produces a realistic spike-like voltage trace
2. Understand the roles of Na, K, and leak currents
3. Prepare this model for later clustering experiments
"""


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Time settings
dt = 0.01
T = 50
steps = int(T / dt)
time = np.linspace(0, T, steps)

# Membrane parameters
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3

E_Na = 50.0
E_K = -77.0
E_L = -54.4

# Starting voltage
V = -65.0

# Initial gating variables
m = 0.05
h = 0.60
n = 0.32


# In[3]:


# Storage for results
V_history = []
m_history = []
h_history = []
n_history = []
I_Na_history = []
I_K_history = []
I_L_history = []


# In[4]:


for i in range(steps):

    # --- Gating variable rate constants ---
    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = 4.0 * np.exp(-(V + 65) / 18)

    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h = 1 / (1 + np.exp(-(V + 35) / 10))

    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = 0.125 * np.exp(-(V + 65) / 80)

    # --- Update gating variables ---
    m += (alpha_m * (1 - m) - beta_m * m) * dt
    h += (alpha_h * (1 - h) - beta_h * h) * dt
    n += (alpha_n * (1 - n) - beta_n * n) * dt

    # --- Currents ---
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)

    # --- External stimulus ---
    I_ext = 10 if 5 <= time[i] < 15 else 0

    # --- Voltage update ---
    V += (-(I_Na + I_K + I_L) + I_ext) / C_m * dt

    # --- Store values ---
    V_history.append(V)
    m_history.append(m)
    h_history.append(h)
    n_history.append(n)
    I_Na_history.append(I_Na)
    I_K_history.append(I_K)
    I_L_history.append(I_L)


# In[5]:


plt.figure(figsize=(10, 5))
plt.plot(time, V_history, label="Membrane Potential (V)")
plt.title("Full Hodgkin-Huxley Style Action Potential")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.grid(True)
plt.legend()
plt.savefig("full_HH_action_potential.png", dpi=300, bbox_inches="tight")
plt.show()


# In[6]:


# Channel distribution parameters
cluster_fraction = 0.5

g_Na_cluster = g_Na * cluster_fraction
g_Na_diffuse = g_Na * (1 - cluster_fraction)


# In[7]:


print("Clustered sodium conductance:", g_Na_cluster)
print("Diffuse sodium conductance:", g_Na_diffuse)
print("Total sodium conductance:", g_Na_cluster + g_Na_diffuse)


# In[8]:


for i in range(steps):

    # --- Gating variable rate constants ---
    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = 4.0 * np.exp(-(V + 65) / 18)

    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h = 1 / (1 + np.exp(-(V + 35) / 10))

    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = 0.125 * np.exp(-(V + 65) / 80)

    # --- Update gating variables ---
    m += (alpha_m * (1 - m) - beta_m * m) * dt
    h += (alpha_h * (1 - h) - beta_h * h) * dt
    n += (alpha_n * (1 - n) - beta_n * n) * dt

    # --- Sodium currents (cluster + diffuse) ---
    I_Na_cluster = g_Na_cluster * (m**3) * h * (V - E_Na)
    I_Na_diffuse = g_Na_diffuse * (m**3) * h * (V - E_Na)

    I_Na_total = I_Na_cluster + I_Na_diffuse

    # --- Other currents ---
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)

    # --- External stimulus ---
    I_ext = 10 if 5 <= time[i] < 6 else 0

    # --- Voltage update ---
    V += (-(I_Na_total + I_K + I_L) + I_ext) / C_m * dt

    # --- Store values ---
    V_history.append(V)
    m_history.append(m)
    h_history.append(h)
    n_history.append(n)
    I_Na_history.append(I_Na_total)
    I_K_history.append(I_K)
    I_L_history.append(I_L)


# In[9]:


# Two-compartment starting conditions
V_cluster = -65.0
V_diffuse = -65.0

# Shared initial gating variables
m_cluster, h_cluster, n_cluster = 0.05, 0.60, 0.32
m_diffuse, h_diffuse, n_diffuse = 0.05, 0.60, 0.32

# Coupling conductance between compartments
g_c = 1.0


# In[10]:


# Storage for two-compartment results
V_cluster_history = []
V_diffuse_history = []


# In[11]:


# Reset time axis for the two-compartment simulation
time = np.linspace(0, T, steps)


# In[12]:


for i in range(steps):

    # --- Rate constants (cluster compartment) ---
    alpha_m_c = 0.1 * (V_cluster + 40) / (1 - np.exp(-(V_cluster + 40) / 10))
    beta_m_c = 4.0 * np.exp(-(V_cluster + 65) / 18)

    alpha_h_c = 0.07 * np.exp(-(V_cluster + 65) / 20)
    beta_h_c = 1 / (1 + np.exp(-(V_cluster + 35) / 10))

    alpha_n_c = 0.01 * (V_cluster + 55) / (1 - np.exp(-(V_cluster + 55) / 10))
    beta_n_c = 0.125 * np.exp(-(V_cluster + 65) / 80)

    # --- Rate constants (diffuse compartment) ---
    alpha_m_d = 0.1 * (V_diffuse + 40) / (1 - np.exp(-(V_diffuse + 40) / 10))
    beta_m_d = 4.0 * np.exp(-(V_diffuse + 65) / 18)

    alpha_h_d = 0.07 * np.exp(-(V_diffuse + 65) / 20)
    beta_h_d = 1 / (1 + np.exp(-(V_diffuse + 35) / 10))

    alpha_n_d = 0.01 * (V_diffuse + 55) / (1 - np.exp(-(V_diffuse + 55) / 10))
    beta_n_d = 0.125 * np.exp(-(V_diffuse + 65) / 80)

    # --- Update gating variables ---
    m_cluster += (alpha_m_c * (1 - m_cluster) - beta_m_c * m_cluster) * dt
    h_cluster += (alpha_h_c * (1 - h_cluster) - beta_h_c * h_cluster) * dt
    n_cluster += (alpha_n_c * (1 - n_cluster) - beta_n_c * n_cluster) * dt

    m_diffuse += (alpha_m_d * (1 - m_diffuse) - beta_m_d * m_diffuse) * dt
    h_diffuse += (alpha_h_d * (1 - h_diffuse) - beta_h_d * h_diffuse) * dt
    n_diffuse += (alpha_n_d * (1 - n_diffuse) - beta_n_d * n_diffuse) * dt

    # --- Currents (cluster compartment) ---
    I_Na_c = g_Na_cluster * (m_cluster**3) * h_cluster * (V_cluster - E_Na)
    I_K_c = g_K * (n_cluster**4) * (V_cluster - E_K)
    I_L_c = g_L * (V_cluster - E_L)

    # --- Currents (diffuse compartment) ---
    I_Na_d = g_Na_diffuse * (m_diffuse**3) * h_diffuse * (V_diffuse - E_Na)
    I_K_d = g_K * (n_diffuse**4) * (V_diffuse - E_K)
    I_L_d = g_L * (V_diffuse - E_L)

    # --- Coupling current ---
    I_couple = g_c * (V_diffuse - V_cluster)

    # --- External stimulus (applied to cluster region) ---
    I_ext = 10 if 5 <= time[i] < 6 else 0

    # --- Voltage updates ---
    V_cluster += (-(I_Na_c + I_K_c + I_L_c) + I_ext + I_couple) / C_m * dt
    V_diffuse += (-(I_Na_d + I_K_d + I_L_d) - I_couple) / C_m * dt

    # --- Store values ---
    V_cluster_history.append(V_cluster)
    V_diffuse_history.append(V_diffuse)


# In[13]:


plt.figure(figsize=(10, 5))
plt.plot(time, V_cluster_history, label="Cluster Voltage")
plt.plot(time, V_diffuse_history, label="Diffuse Voltage")
plt.title("Two-Compartment Model: Cluster vs Diffuse Voltage")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid(True)
plt.savefig("two_compartment_voltage.png", dpi=300, bbox_inches="tight")
plt.show()


# In[14]:


# Reset two-compartment histories
V_cluster_history = []
V_diffuse_history = []


# In[15]:


# Function to test if spike occurs
def check_spike(V_trace):
    return any(v > 0 for v in V_trace)


# In[16]:


def run_two_compartment_simulation(cluster_fraction, T=50, dt=0.01):
    steps = int(T / dt)
    time = np.linspace(0, T, steps)

    # Fixed total sodium conductance, redistributed by cluster_fraction
    g_Na_total = 120.0
    g_Na_cluster = g_Na_total * cluster_fraction
    g_Na_diffuse = g_Na_total * (1 - cluster_fraction)

    # Other membrane parameters
    C_m = 1.0
    g_K = 36.0
    g_L = 0.3
    g_c = 1.0

    E_Na = 50.0
    E_K = -77.0
    E_L = -54.4

    # Initial voltages
    V_cluster = -65.0
    V_diffuse = -65.0

    # Initial gating variables
    m_cluster, h_cluster, n_cluster = 0.05, 0.60, 0.32
    m_diffuse, h_diffuse, n_diffuse = 0.05, 0.60, 0.32

    # Storage
    V_cluster_history = []
    V_diffuse_history = []

    for i in range(steps):
        # Cluster rates
        alpha_m_c = 0.1 * (V_cluster + 40) / (1 - np.exp(-(V_cluster + 40) / 10))
        beta_m_c = 4.0 * np.exp(-(V_cluster + 65) / 18)

        alpha_h_c = 0.07 * np.exp(-(V_cluster + 65) / 20)
        beta_h_c = 1 / (1 + np.exp(-(V_cluster + 35) / 10))

        alpha_n_c = 0.01 * (V_cluster + 55) / (1 - np.exp(-(V_cluster + 55) / 10))
        beta_n_c = 0.125 * np.exp(-(V_cluster + 65) / 80)

        # Diffuse rates
        alpha_m_d = 0.1 * (V_diffuse + 40) / (1 - np.exp(-(V_diffuse + 40) / 10))
        beta_m_d = 4.0 * np.exp(-(V_diffuse + 65) / 18)

        alpha_h_d = 0.07 * np.exp(-(V_diffuse + 65) / 20)
        beta_h_d = 1 / (1 + np.exp(-(V_diffuse + 35) / 10))

        alpha_n_d = 0.01 * (V_diffuse + 55) / (1 - np.exp(-(V_diffuse + 55) / 10))
        beta_n_d = 0.125 * np.exp(-(V_diffuse + 65) / 80)

        # Update gating variables
        m_cluster += (alpha_m_c * (1 - m_cluster) - beta_m_c * m_cluster) * dt
        h_cluster += (alpha_h_c * (1 - h_cluster) - beta_h_c * h_cluster) * dt
        n_cluster += (alpha_n_c * (1 - n_cluster) - beta_n_c * n_cluster) * dt

        m_diffuse += (alpha_m_d * (1 - m_diffuse) - beta_m_d * m_diffuse) * dt
        h_diffuse += (alpha_h_d * (1 - h_diffuse) - beta_h_d * h_diffuse) * dt
        n_diffuse += (alpha_n_d * (1 - n_diffuse) - beta_n_d * n_diffuse) * dt

        # Currents
        I_Na_c = g_Na_cluster * (m_cluster**3) * h_cluster * (V_cluster - E_Na)
        I_K_c = g_K * (n_cluster**4) * (V_cluster - E_K)
        I_L_c = g_L * (V_cluster - E_L)

        I_Na_d = g_Na_diffuse * (m_diffuse**3) * h_diffuse * (V_diffuse - E_Na)
        I_K_d = g_K * (n_diffuse**4) * (V_diffuse - E_K)
        I_L_d = g_L * (V_diffuse - E_L)

        I_couple = g_c * (V_diffuse - V_cluster)

        # Stimulus to cluster compartment only
        I_ext = 20 if 5 <= time[i] < 7 else 0

        # Voltage updates
        V_cluster += (-(I_Na_c + I_K_c + I_L_c) + I_ext + I_couple) / C_m * dt
        V_diffuse += (-(I_Na_d + I_K_d + I_L_d) - I_couple) / C_m * dt

        # Store
        V_cluster_history.append(V_cluster)
        V_diffuse_history.append(V_diffuse)

    return time, V_cluster_history, V_diffuse_history


# In[17]:


time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=0.5)


# In[18]:


plt.figure(figsize=(10, 5))
plt.plot(time, V_c, label="Cluster Voltage")
plt.plot(time, V_d, label="Diffuse Voltage")
plt.title("Two-Compartment Simulation")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid(True)
plt.show()


# In[19]:


time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=0.1)


# In[20]:


plt.figure(figsize=(10, 5))
plt.plot(time, V_c, label="Cluster Voltage")
plt.plot(time, V_d, label="Diffuse Voltage")
plt.title("Two-Compartment Simulation")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


plt.figure(figsize=(10, 5))
plt.plot(time, V_c, label="Cluster Voltage")
plt.plot(time, V_d, label="Diffuse Voltage")
plt.title("Two-Compartment Simulation")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid(True)
plt.show()


# In[22]:


print("Cluster spike success:", check_spike(V_c))
print("Diffuse spike success:", check_spike(V_d))
print("Cluster max voltage:", max(V_c))
print("Diffuse max voltage:", max(V_d))


# In[23]:


plt.figure(figsize=(10, 5))
plt.plot(time, V_c, label="Cluster Voltage")
plt.plot(time, V_d, label="Diffuse Voltage")
plt.title("Two-Compartment Simulation")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid(True)
plt.show()


# In[24]:


print("Cluster spike success:", check_spike(V_c))
print("Diffuse spike success:", check_spike(V_d))
print("Cluster max voltage:", max(V_c))
print("Diffuse max voltage:", max(V_d))


# In[25]:


cluster_values = np.linspace(0, 1, 10)
cluster_success = []
diffuse_success = []

for f in cluster_values:
    time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=f)

    cluster_success.append(check_spike(V_c))
    diffuse_success.append(check_spike(V_d))


# In[26]:


plt.figure(figsize=(8, 5))
plt.plot(cluster_values, cluster_success, marker='o', label="Cluster Success")
plt.plot(cluster_values, diffuse_success, marker='s', label="Diffuse Success")
plt.xlabel("Cluster Fraction")
plt.ylabel("Spike Success (True=1, False=0)")
plt.title("Spike Success vs Channel Clustering")
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


cluster_values = np.linspace(0.3, 0.8, 50)
cluster_success = []
diffuse_success = []

for f in cluster_values:
    time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=f)
    cluster_success.append(int(check_spike(V_c)))
    diffuse_success.append(int(check_spike(V_d)))


# In[28]:


plt.figure(figsize=(8, 5))
plt.plot(cluster_values, cluster_success, marker='o', label="Cluster Success")
plt.plot(cluster_values, diffuse_success, marker='s', label="Diffuse Success")
plt.xlabel("Cluster Fraction")
plt.ylabel("Spike Success (1 = spike, 0 = no spike)")
plt.title("Spike Success vs Channel Clustering")
plt.legend()
plt.grid(True)
plt.show()


# In[29]:


cluster_max = []
diffuse_max = []

for f in cluster_values:
    time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=f)
    cluster_max.append(max(V_c))
    diffuse_max.append(max(V_d))


# In[30]:


both_spike_indices = []
mismatch_values = []

for i in range(len(cluster_values)):
    if cluster_success[i] == 1 and diffuse_success[i] == 1:
        both_spike_indices.append(i)
        mismatch = abs(cluster_max[i] - diffuse_max[i])
        mismatch_values.append(mismatch)

if len(both_spike_indices) > 0:
    best_local_index = mismatch_values.index(min(mismatch_values))
    best_global_index = both_spike_indices[best_local_index]

    optimal_cluster_fraction = cluster_values[best_global_index]
    optimal_cluster_max = cluster_max[best_global_index]
    optimal_diffuse_max = diffuse_max[best_global_index]
    optimal_mismatch = mismatch_values[best_local_index]

    print("Optimal cluster_fraction:", optimal_cluster_fraction)
    print("Cluster max voltage at optimum:", optimal_cluster_max)
    print("Diffuse max voltage at optimum:", optimal_diffuse_max)
    print("Voltage mismatch at optimum:", optimal_mismatch)
else:
    print("No cluster_fraction values found where both compartments spike.")


# In[31]:


plt.figure(figsize=(8, 5))
plt.plot(cluster_values, cluster_max, marker='o', label="Cluster Max Voltage")
plt.plot(cluster_values, diffuse_max, marker='s', label="Diffuse Max Voltage")
plt.axhline(0, linestyle='--', label="Spike Threshold (0 mV)")
plt.xlabel("Cluster Fraction")
plt.ylabel("Max Membrane Potential (mV)")
plt.title("Max Voltage vs Channel Clustering")
plt.legend()
plt.grid(True)
plt.show()


# In[32]:


valid_cluster_values = [cluster_values[i] for i in both_spike_indices]

plt.figure(figsize=(8, 5))
plt.plot(valid_cluster_values, mismatch_values, marker='o')
plt.xlabel("Cluster Fraction")
plt.ylabel("Mismatch (mV)")
plt.title("Mismatch vs Channel Clustering (Dual-Spiking Region)")
plt.grid(True)
plt.savefig("Mismatch vs Channel Clustering (Dual-Spiking Region).png", dpi=300, bbox_inches="tight")
plt.show()


# In[33]:


def first_spike_time(time, V, threshold=0):
    for t, v in zip(time, V):
        if v > threshold:
            return t
    return None


# In[34]:


delay_values = []
valid_delay_cluster_values = []

for f in cluster_values:
    time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=f)

    t_cluster = first_spike_time(time, V_c, threshold=0)
    t_diffuse = first_spike_time(time, V_d, threshold=0)

    if t_cluster is not None and t_diffuse is not None:
        delay = t_diffuse - t_cluster
        delay_values.append(delay)
        valid_delay_cluster_values.append(f)


# In[35]:


plt.figure(figsize=(8, 5))
plt.plot(valid_delay_cluster_values, delay_values, marker='o')
plt.axhline(0, linestyle='--')
plt.xlabel("Cluster Fraction")
plt.ylabel("Spike Delay (ms)")
plt.title("Intercompartment Spike Delay vs Channel Clustering")
plt.grid(True)
plt.savefig("Intercompartment Spike Delay vs Channel Clustering.png", dpi=300, bbox_inches="tight")
plt.show()


# In[36]:


tradeoff_mismatch = []
tradeoff_delay = []
tradeoff_cluster_values = []

for f in cluster_values:
    time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=f)

    # Spike times
    t_cluster = first_spike_time(time, V_c, threshold=0)
    t_diffuse = first_spike_time(time, V_d, threshold=0)

    # Max voltages
    Vc_max = max(V_c)
    Vd_max = max(V_d)

    # Only include points where both spike
    if t_cluster is not None and t_diffuse is not None:
        delay = t_diffuse - t_cluster
        mismatch = abs(Vc_max - Vd_max)

        tradeoff_delay.append(delay)
        tradeoff_mismatch.append(mismatch)
        tradeoff_cluster_values.append(f)


# In[37]:


plt.figure(figsize=(8, 5))
plt.scatter(tradeoff_mismatch, tradeoff_delay)

plt.xlabel("Mismatch |Vmax_cluster - Vmax_diffuse| (mV)")
plt.ylabel("Spike Delay (Diffuse - Cluster) (ms)")
plt.title("Tradeoff Between Spike Coordination and Propagation Speed")

plt.grid(True)
plt.savefig("Tradeoff Between Spike Coordination and Propagation Speed.png", dpi=300, bbox_inches="tight")
plt.show()


# In[38]:


plt.figure(figsize=(8, 5))
sc = plt.scatter(tradeoff_mismatch, tradeoff_delay, c=tradeoff_cluster_values)

plt.xlabel("Mismatch (mV)")
plt.ylabel("Delay (ms)")
plt.title("Tradeoff Between Coordination and Propagation Delay")

plt.colorbar(sc, label="Cluster Fraction")
plt.grid(True)
plt.savefig("Tradeoff between coordination and propagation delay.png", dpi=300, bbox_inches="tight")
plt.show()


# In[39]:


feasible_cluster_values = []

for f in cluster_values:
    time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=f)

    t_cluster = first_spike_time(time, V_c, threshold=0)
    t_diffuse = first_spike_time(time, V_d, threshold=0)

    # Check both compartments spike
    if t_cluster is not None and t_diffuse is not None:
        delay = t_diffuse - t_cluster

        # Apply delay constraint
        if delay < 0.5:
            feasible_cluster_values.append(f)

# Print result
if feasible_cluster_values:
    print("Feasible cluster_fraction range:")
    print(min(feasible_cluster_values), "to", max(feasible_cluster_values))
else:
    print("No feasible region found.")


# In[40]:


def run_two_compartment_simulation(
    cluster_fraction,
    T=50,
    dt=0.01,
    g_Na_total=120.0,
    g_c=1.0,
    I_ext_amp=20.0
):
    steps = int(T / dt)
    time = np.linspace(0, T, steps)

    # Fixed total sodium conductance, redistributed by cluster_fraction
    g_Na_cluster = g_Na_total * cluster_fraction
    g_Na_diffuse = g_Na_total * (1 - cluster_fraction)

    # Other membrane parameters
    C_m = 1.0
    g_K = 36.0
    g_L = 0.3

    E_Na = 50.0
    E_K = -77.0
    E_L = -54.4

    # Initial voltages
    V_cluster = -65.0
    V_diffuse = -65.0

    # Initial gating variables
    m_cluster, h_cluster, n_cluster = 0.05, 0.60, 0.32
    m_diffuse, h_diffuse, n_diffuse = 0.05, 0.60, 0.32

    # Storage
    V_cluster_history = []
    V_diffuse_history = []

    for i in range(steps):
        # Cluster rates
        alpha_m_c = 0.1 * (V_cluster + 40) / (1 - np.exp(-(V_cluster + 40) / 10))
        beta_m_c = 4.0 * np.exp(-(V_cluster + 65) / 18)

        alpha_h_c = 0.07 * np.exp(-(V_cluster + 65) / 20)
        beta_h_c = 1 / (1 + np.exp(-(V_cluster + 35) / 10))

        alpha_n_c = 0.01 * (V_cluster + 55) / (1 - np.exp(-(V_cluster + 55) / 10))
        beta_n_c = 0.125 * np.exp(-(V_cluster + 65) / 80)

        # Diffuse rates
        alpha_m_d = 0.1 * (V_diffuse + 40) / (1 - np.exp(-(V_diffuse + 40) / 10))
        beta_m_d = 4.0 * np.exp(-(V_diffuse + 65) / 18)

        alpha_h_d = 0.07 * np.exp(-(V_diffuse + 65) / 20)
        beta_h_d = 1 / (1 + np.exp(-(V_diffuse + 35) / 10))

        alpha_n_d = 0.01 * (V_diffuse + 55) / (1 - np.exp(-(V_diffuse + 55) / 10))
        beta_n_d = 0.125 * np.exp(-(V_diffuse + 65) / 80)

        # Update gating variables
        m_cluster += (alpha_m_c * (1 - m_cluster) - beta_m_c * m_cluster) * dt
        h_cluster += (alpha_h_c * (1 - h_cluster) - beta_h_c * h_cluster) * dt
        n_cluster += (alpha_n_c * (1 - n_cluster) - beta_n_c * n_cluster) * dt

        m_diffuse += (alpha_m_d * (1 - m_diffuse) - beta_m_d * m_diffuse) * dt
        h_diffuse += (alpha_h_d * (1 - h_diffuse) - beta_h_d * h_diffuse) * dt
        n_diffuse += (alpha_n_d * (1 - n_diffuse) - beta_n_d * n_diffuse) * dt

        # Currents
        I_Na_c = g_Na_cluster * (m_cluster**3) * h_cluster * (V_cluster - E_Na)
        I_K_c = g_K * (n_cluster**4) * (V_cluster - E_K)
        I_L_c = g_L * (V_cluster - E_L)

        I_Na_d = g_Na_diffuse * (m_diffuse**3) * h_diffuse * (V_diffuse - E_Na)
        I_K_d = g_K * (n_diffuse**4) * (V_diffuse - E_K)
        I_L_d = g_L * (V_diffuse - E_L)

        I_couple = g_c * (V_diffuse - V_cluster)

        # Stimulus to cluster compartment only
        I_ext = I_ext_amp if 5 <= time[i] < 7 else 0

        # Voltage updates
        V_cluster += (-(I_Na_c + I_K_c + I_L_c) + I_ext + I_couple) / C_m * dt
        V_diffuse += (-(I_Na_d + I_K_d + I_L_d) - I_couple) / C_m * dt

        # Store
        V_cluster_history.append(V_cluster)
        V_diffuse_history.append(V_diffuse)

    return time, V_cluster_history, V_diffuse_history


# In[41]:


def check_spike(V, threshold=0):
    return max(V) > threshold


def first_spike_time(time, V, threshold=0):
    for t, v in zip(time, V):
        if v > threshold:
            return t
    return None


def analyze_parameter_setting(
    cluster_values,
    g_c=1.0,
    g_Na_total=120.0,
    I_ext_amp=20.0,
    delay_threshold=0.5
):
    cluster_success = []
    diffuse_success = []
    cluster_max = []
    diffuse_max = []
    delay_values = []
    valid_delay_cluster_values = []

    both_spike_indices = []
    mismatch_values = []

    tradeoff_mismatch = []
    tradeoff_delay = []
    tradeoff_cluster_values = []

    feasible_cluster_values = []

    for i, f in enumerate(cluster_values):
        time, V_c, V_d = run_two_compartment_simulation(
            cluster_fraction=f,
            g_c=g_c,
            g_Na_total=g_Na_total,
            I_ext_amp=I_ext_amp
        )

        cluster_spike = int(check_spike(V_c))
        diffuse_spike = int(check_spike(V_d))

        cluster_success.append(cluster_spike)
        diffuse_success.append(diffuse_spike)

        Vc_max = max(V_c)
        Vd_max = max(V_d)

        cluster_max.append(Vc_max)
        diffuse_max.append(Vd_max)

        t_cluster = first_spike_time(time, V_c, threshold=0)
        t_diffuse = first_spike_time(time, V_d, threshold=0)

        if t_cluster is not None and t_diffuse is not None:
            delay = t_diffuse - t_cluster
            mismatch = abs(Vc_max - Vd_max)

            delay_values.append(delay)
            valid_delay_cluster_values.append(f)

            both_spike_indices.append(i)
            mismatch_values.append(mismatch)

            tradeoff_delay.append(delay)
            tradeoff_mismatch.append(mismatch)
            tradeoff_cluster_values.append(f)

            if delay < delay_threshold:
                feasible_cluster_values.append(f)

    # amplitude optimum
    if len(both_spike_indices) > 0:
        best_local_index = mismatch_values.index(min(mismatch_values))
        best_global_index = both_spike_indices[best_local_index]

        optimal_cluster_fraction = cluster_values[best_global_index]
        optimal_cluster_max = cluster_max[best_global_index]
        optimal_diffuse_max = diffuse_max[best_global_index]
        optimal_mismatch = mismatch_values[best_local_index]
    else:
        optimal_cluster_fraction = None
        optimal_cluster_max = None
        optimal_diffuse_max = None
        optimal_mismatch = None

    # feasible region
    if len(feasible_cluster_values) > 0:
        feasible_min = min(feasible_cluster_values)
        feasible_max = max(feasible_cluster_values)
    else:
        feasible_min = None
        feasible_max = None

    return {
        "cluster_values": cluster_values,
        "cluster_success": cluster_success,
        "diffuse_success": diffuse_success,
        "cluster_max": cluster_max,
        "diffuse_max": diffuse_max,
        "delay_values": delay_values,
        "valid_delay_cluster_values": valid_delay_cluster_values,
        "both_spike_indices": both_spike_indices,
        "mismatch_values": mismatch_values,
        "tradeoff_mismatch": tradeoff_mismatch,
        "tradeoff_delay": tradeoff_delay,
        "tradeoff_cluster_values": tradeoff_cluster_values,
        "feasible_cluster_values": feasible_cluster_values,
        "optimal_cluster_fraction": optimal_cluster_fraction,
        "optimal_cluster_max": optimal_cluster_max,
        "optimal_diffuse_max": optimal_diffuse_max,
        "optimal_mismatch": optimal_mismatch,
        "feasible_min": feasible_min,
        "feasible_max": feasible_max,
        "g_c": g_c,
        "g_Na_total": g_Na_total,
        "I_ext_amp": I_ext_amp,
        "delay_threshold": delay_threshold
    }


# In[42]:


def plot_analysis_results_combined(results, title_suffix="", save_name=None):
    cluster_values = results["cluster_values"]
    cluster_success = results["cluster_success"]
    diffuse_success = results["diffuse_success"]
    cluster_max = results["cluster_max"]
    diffuse_max = results["diffuse_max"]
    both_spike_indices = results["both_spike_indices"]
    mismatch_values = results["mismatch_values"]
    delay_values = results["delay_values"]
    valid_delay_cluster_values = results["valid_delay_cluster_values"]
    tradeoff_mismatch = results["tradeoff_mismatch"]
    tradeoff_delay = results["tradeoff_delay"]
    tradeoff_cluster_values = results["tradeoff_cluster_values"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    axs = axs.flatten()

    # 1. Spike success
    axs[0].plot(cluster_values, cluster_success, marker='o', label="Cluster")
    axs[0].plot(cluster_values, diffuse_success, marker='s', label="Diffuse")
    axs[0].set_title("Spike Success")
    axs[0].set_xlabel("Cluster Fraction")
    axs[0].set_ylabel("Spike Success")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Max voltage
    axs[1].plot(cluster_values, cluster_max, marker='o', label="Cluster")
    axs[1].plot(cluster_values, diffuse_max, marker='s', label="Diffuse")
    axs[1].axhline(0, linestyle='--', label="Spike Threshold (0 mV)")
    axs[1].set_title("Max Voltage")
    axs[1].set_xlabel("Cluster Fraction")
    axs[1].set_ylabel("Max Voltage (mV)")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Mismatch curve
    valid_cluster_values = [cluster_values[i] for i in both_spike_indices]
    axs[2].plot(valid_cluster_values, mismatch_values, marker='o')
    axs[2].set_title("Mismatch")
    axs[2].set_xlabel("Cluster Fraction")
    axs[2].set_ylabel("|Vmax_cluster - Vmax_diffuse|")
    axs[2].grid(True)

    # 4. Delay curve
    axs[3].plot(valid_delay_cluster_values, delay_values, marker='o')
    axs[3].axhline(0, linestyle='--')
    axs[3].set_title("Delay")
    axs[3].set_xlabel("Cluster Fraction")
    axs[3].set_ylabel("Spike Delay (ms)")
    axs[3].grid(True)

    # 5. Tradeoff plot
    sc = axs[4].scatter(tradeoff_mismatch, tradeoff_delay, c=tradeoff_cluster_values)
    axs[4].set_title("Tradeoff: Coordination vs Speed")
    axs[4].set_xlabel("Mismatch |Vmax_cluster - Vmax_diffuse| (mV)")
    axs[4].set_ylabel("Spike Delay (Diffuse - Cluster) (ms)")
    axs[4].grid(True)

    cbar = fig.colorbar(sc, ax=axs[4])
    cbar.set_label("Cluster Fraction")

    # Remove empty 6th panel
    fig.delaxes(axs[5])

    fig.suptitle(f"Analysis Summary {title_suffix}", fontsize=16)
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.show()


def print_summary(results, label="Condition"):
    print(f"--- {label} ---")
    print("g_c =", results["g_c"])
    print("g_Na_total =", results["g_Na_total"])
    print("I_ext_amp =", results["I_ext_amp"])
    print("delay_threshold =", results["delay_threshold"])
    print()

    print("Amplitude optimum cluster_fraction:", results["optimal_cluster_fraction"])
    print("Cluster max voltage at optimum:", results["optimal_cluster_max"])
    print("Diffuse max voltage at optimum:", results["optimal_diffuse_max"])
    print("Voltage mismatch at optimum:", results["optimal_mismatch"])
    print()

    if results["feasible_min"] is not None:
        print("Feasible cluster_fraction range:")
        print(results["feasible_min"], "to", results["feasible_max"])
    else:
        print("No feasible region found.")
    print()


# In[43]:


cluster_values = np.linspace(0.0, 1.0, 100)
delay_threshold = 0.5

baseline_results = analyze_parameter_setting(
    cluster_values=cluster_values,
    g_c=1.0,
    g_Na_total=120.0,
    I_ext_amp=20.0,
    delay_threshold=delay_threshold
)

print_summary(baseline_results, label="Baseline")
plot_analysis_results_combined(
    baseline_results,
    title_suffix="(Baseline)",
    save_name="baseline_analysis_summary.png"
)


# In[44]:


coupling_conditions = {
    "Low coupling": 0.5,
    "Baseline coupling": 1.0,
    "High coupling": 1.5
}

coupling_results = {}

for label, g_c_val in coupling_conditions.items():
    results = analyze_parameter_setting(
        cluster_values=cluster_values,
        g_c=g_c_val,
        g_Na_total=120.0,
        I_ext_amp=20.0,
        delay_threshold=delay_threshold
    )
    coupling_results[label] = results
    print_summary(results, label=label)


# In[45]:


for label, results in coupling_results.items():
    safe_label = label.lower().replace(" ", "_")
    plot_analysis_results_combined(
        results,
        title_suffix=f"({label})",
        save_name=f"{safe_label}_analysis_summary.png"
    )


# In[46]:


plt.figure(figsize=(8, 5))

for label, results in coupling_results.items():
    valid_cluster_values = [results["cluster_values"][i] for i in results["both_spike_indices"]]
    plt.plot(valid_cluster_values, results["mismatch_values"], marker='o', label=label)

plt.xlabel("Cluster Fraction")
plt.ylabel("Mismatch |Vmax_cluster - Vmax_diffuse|")
plt.title("Mismatch Curves Across Coupling Conductance")
plt.legend()
plt.grid(True)
plt.savefig("coupling_mismatch_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# In[47]:


plt.figure(figsize=(8, 5))

for label, results in coupling_results.items():
    plt.plot(
        results["valid_delay_cluster_values"],
        results["delay_values"],
        marker='o',
        label=label
    )

plt.axhline(0, linestyle='--')
plt.xlabel("Cluster Fraction")
plt.ylabel("Spike Delay (Diffuse - Cluster) [ms]")
plt.title("Delay Curves Across Coupling Conductance")
plt.legend()
plt.grid(True)
plt.savefig("coupling_delay_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# In[48]:


sodium_conditions = {
    "Low total sodium": 100.0,
    "Baseline total sodium": 120.0,
    "High total sodium": 140.0
}

sodium_results = {}

for label, gNa_val in sodium_conditions.items():
    results = analyze_parameter_setting(
        cluster_values=cluster_values,
        g_c=1.0,
        g_Na_total=gNa_val,
        I_ext_amp=20.0,
        delay_threshold=delay_threshold
    )
    sodium_results[label] = results
    print_summary(results, label=label)


# In[49]:


for label, results in sodium_results.items():
    safe_label = label.lower().replace(" ", "_")
    plot_analysis_results_combined(
        results,
        title_suffix=f"({label})",
        save_name=f"{safe_label}_analysis_summary.png"
    )


# In[50]:


plt.figure(figsize=(8, 5))

for label, results in sodium_results.items():
    valid_cluster_values = [results["cluster_values"][i] for i in results["both_spike_indices"]]
    plt.plot(valid_cluster_values, results["mismatch_values"], marker='o', label=label)

plt.xlabel("Cluster Fraction")
plt.ylabel("Mismatch |Vmax_cluster - Vmax_diffuse|")
plt.title("Mismatch Curves Across Total Sodium Conductance")
plt.legend()
plt.grid(True)
plt.savefig("sodium_mismatch_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# In[51]:


plt.figure(figsize=(8, 5))

for label, results in sodium_results.items():
    plt.plot(
        results["valid_delay_cluster_values"],
        results["delay_values"],
        marker='o',
        label=label
    )

plt.axhline(0, linestyle='--')
plt.xlabel("Cluster Fraction")
plt.ylabel("Spike Delay (Diffuse - Cluster) [ms]")
plt.title("Delay Curves Across Total Sodium Conductance")
plt.legend()
plt.grid(True)
plt.savefig("sodium_delay_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# In[52]:


input_conditions = {
    "Low input": 15.0,
    "Baseline input": 20.0,
    "High input": 25.0
}

input_results = {}

for label, I_val in input_conditions.items():
    results = analyze_parameter_setting(
        cluster_values=cluster_values,
        g_c=1.0,
        g_Na_total=120.0,
        I_ext_amp=I_val,
        delay_threshold=delay_threshold
    )
    input_results[label] = results
    print_summary(results, label=label)


# In[53]:


for label, results in input_results.items():
    safe_label = label.lower().replace(" ", "_")
    plot_analysis_results_combined(
        results,
        title_suffix=f"({label})",
        save_name=f"{safe_label}_analysis_summary.png"
    )


# In[54]:


plt.figure(figsize=(8, 5))

for label, results in input_results.items():
    valid_cluster_values = [results["cluster_values"][i] for i in results["both_spike_indices"]]
    plt.plot(valid_cluster_values, results["mismatch_values"], marker='o', label=label)

plt.xlabel("Cluster Fraction")
plt.ylabel("Mismatch |Vmax_cluster - Vmax_diffuse|")
plt.title("Mismatch Curves Across Input Strength")
plt.legend()
plt.grid(True)
plt.savefig("input_mismatch_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# In[55]:


plt.figure(figsize=(8, 5))

for label, results in input_results.items():
    plt.plot(
        results["valid_delay_cluster_values"],
        results["delay_values"],
        marker='o',
        label=label
    )

plt.axhline(0, linestyle='--')
plt.xlabel("Cluster Fraction")
plt.ylabel("Spike Delay (Diffuse - Cluster) [ms]")
plt.title("Delay Curves Across Input Strength")
plt.legend()
plt.grid(True)
plt.savefig("input_delay_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# In[56]:


phase1_summary = []

for group_name, result_dict in [
    ("Coupling", coupling_results),
    ("Total Sodium", sodium_results),
    ("Input Strength", input_results)
]:
    for label, results in result_dict.items():
        phase1_summary.append({
            "Group": group_name,
            "Condition": label,
            "g_c": results["g_c"],
            "g_Na_total": results["g_Na_total"],
            "I_ext_amp": results["I_ext_amp"],
            "Amplitude optimum": results["optimal_cluster_fraction"],
            "Optimal mismatch": results["optimal_mismatch"],
            "Feasible min": results["feasible_min"],
            "Feasible max": results["feasible_max"]
        })

import pandas as pd
phase1_summary_df = pd.DataFrame(phase1_summary)
phase1_summary_df


# In[57]:


regime_labels = []

for c_spike, d_spike in zip(
    baseline_results["cluster_success"],
    baseline_results["diffuse_success"]
):
    if c_spike == 0 and d_spike == 1:
        regime_labels.append("Diffuse only")
    elif c_spike == 1 and d_spike == 1:
        regime_labels.append("Both spike")
    elif c_spike == 1 and d_spike == 0:
        regime_labels.append("Cluster only")
    else:
        regime_labels.append("Neither")


# In[58]:


regime_to_y = {
    "Diffuse only": 1,
    "Both spike": 2,
    "Cluster only": 3,
    "Neither": 0
}

regime_y = [regime_to_y[r] for r in regime_labels]


# In[59]:


plt.figure(figsize=(10, 5))

# Regime points
plt.scatter(cluster_values, regime_y, s=70)

# Shade feasible timing region
if baseline_results["feasible_min"] is not None and baseline_results["feasible_max"] is not None:
    plt.axvspan(
        baseline_results["feasible_min"],
        baseline_results["feasible_max"],
        alpha=0.2,
        label="Feasible timing region"
    )

# Mark amplitude optimum
plt.axvline(
    baseline_results["optimal_cluster_fraction"],
    linestyle='--',
    linewidth=2,
    label="Amplitude optimum"
)

plt.yticks(
    [0, 1, 2, 3],
    ["Neither", "Diffuse only", "Both spike", "Cluster only"]
)

plt.xlabel("Cluster Fraction")
plt.ylabel("Operating Regime")
plt.title("Regime / Operating Map")
plt.grid(True, axis='x')
plt.legend()
plt.savefig("baseline_regime_operating_map.png", dpi=300, bbox_inches="tight")
plt.show()


# In[60]:


def compute_optimum_feasible_gap(results):
    opt = results["optimal_cluster_fraction"]
    feasible_max = results["feasible_max"]

    if opt is None or feasible_max is None:
        return None

    return opt - feasible_max


# In[61]:


gap_summary = []

for group_name, result_dict in [
    ("Coupling", coupling_results),
    ("Total Sodium", sodium_results),
    ("Input Strength", input_results)
]:
    for label, results in result_dict.items():
        gap = compute_optimum_feasible_gap(results)

        gap_summary.append({
            "Group": group_name,
            "Condition": label,
            "g_c": results["g_c"],
            "g_Na_total": results["g_Na_total"],
            "I_ext_amp": results["I_ext_amp"],
            "Amplitude optimum": results["optimal_cluster_fraction"],
            "Feasible max": results["feasible_max"],
            "Gap (optimum - feasible max)": gap
        })

gap_summary_df = pd.DataFrame(gap_summary)
gap_summary_df


# In[62]:


gap_summary_df.sort_values(["Group", "Gap (optimum - feasible max)"])


# In[63]:


interaction_conditions = {
    "Low coupling + Low input": {"g_c": 0.5, "I_ext_amp": 15.0},
    "Low coupling + High input": {"g_c": 0.5, "I_ext_amp": 25.0},
    "High coupling + Low input": {"g_c": 1.5, "I_ext_amp": 15.0},
    "High coupling + High input": {"g_c": 1.5, "I_ext_amp": 25.0}
}

interaction_results = {}

for label, params in interaction_conditions.items():
    results = analyze_parameter_setting(
        cluster_values=cluster_values,
        g_c=params["g_c"],
        g_Na_total=120.0,
        I_ext_amp=params["I_ext_amp"],
        delay_threshold=delay_threshold
    )
    interaction_results[label] = results
    print_summary(results, label=label)


# In[64]:


interaction_summary = []

for label, results in interaction_results.items():
    gap = compute_optimum_feasible_gap(results)

    interaction_summary.append({
        "Condition": label,
        "g_c": results["g_c"],
        "I_ext_amp": results["I_ext_amp"],
        "Amplitude optimum": results["optimal_cluster_fraction"],
        "Feasible min": results["feasible_min"],
        "Feasible max": results["feasible_max"],
        "Gap (optimum - feasible max)": gap
    })

interaction_summary_df = pd.DataFrame(interaction_summary)
interaction_summary_df


# In[65]:


interaction_summary_df.sort_values("Gap (optimum - feasible max)")


# In[66]:


plot_df = interaction_summary_df.copy()

plt.figure(figsize=(8, 5))
plt.bar(plot_df["Condition"], plot_df["Gap (optimum - feasible max)"])
plt.axhline(0, linestyle='--')
plt.ylabel("Gap (optimum - feasible max)")
plt.title("2x2 Coupling × Input Interaction")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis='y')
plt.savefig("interaction_gap_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# In[67]:


for label, results in interaction_results.items():
    safe_label = label.lower().replace(" ", "_").replace("+", "plus")
    plot_analysis_results_combined(
        results,
        title_suffix=f"({label})",
        save_name=f"{safe_label}_analysis_summary.png"
    )


# In[68]:


import numpy as np
import matplotlib.pyplot as plt


def add_feasible_region_axis(ax, result_dict, title, y_labels):
    y_positions = np.arange(len(result_dict))

    for y, (label, results) in zip(y_positions, result_dict.items()):
        feasible_min = results["feasible_min"]
        feasible_max = results["feasible_max"]
        optimum = results["optimal_cluster_fraction"]

        # Faint full clustering range
        ax.hlines(
            y=y,
            xmin=0,
            xmax=1,
            linewidth=1,
            color="#1f77b4",
            alpha=0.08
        )

        # Feasible operating region
        if feasible_min is not None and feasible_max is not None:
            ax.hlines(
                y=y,
                xmin=feasible_min,
                xmax=feasible_max,
                linewidth=10,
                color="#1f77b4",
                alpha=0.95,
                label="Feasible Region" if y == 0 else None
            )

        # Mismatch minimum
        if optimum is not None:
            ax.plot(
                optimum,
                y,
                marker="o",
                markersize=9,
                color="black",
                markeredgecolor="white",
                markeredgewidth=1.8,
                linestyle="None",
                label="Mismatch Minimum" if y == 0 else None
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.25)


fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.5), sharex=True)

add_feasible_region_axis(
    axes[0],
    coupling_results,
    "A. Coupling Strength",
    [
        r"$g_c = 0.5$",
        r"$g_c = 1.0$",
        r"$g_c = 1.5$"
    ]
)

add_feasible_region_axis(
    axes[1],
    sodium_results,
    "B. Total Sodium Conductance",
    [
        r"$g_{\mathrm{Na,total}} = 100$ mS/cm$^2$",
        r"$g_{\mathrm{Na,total}} = 120$ mS/cm$^2$",
        r"$g_{\mathrm{Na,total}} = 140$ mS/cm$^2$"
    ]
)

add_feasible_region_axis(
    axes[2],
    input_results,
    "C. Input Strength",
    [
        r"$I_0 = 15$ $\mu$A/cm$^2$",
        r"$I_0 = 20$ $\mu$A/cm$^2$",
        r"$I_0 = 25$ $\mu$A/cm$^2$"
    ]
)

axes[-1].set_xlabel(
    "Sodium Channel Clustering Fraction (α)",
    fontsize=12,
    labelpad=8
)

for ax in axes[:-1]:
    ax.set_xlabel("")


axes[0].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.60),
    frameon=False,
    fontsize=11
)


plt.tight_layout(rect=[0, 0, 0.80, 0.94], h_pad=2.0)

plt.savefig("Fig5_parameter_sweep_feasible_regions.png", dpi=300, bbox_inches="tight")
plt.savefig("Fig5_parameter_sweep_feasible_regions.svg", bbox_inches="tight")

plt.show()


# In[69]:


# ---------- Supplementary Figure S1 ----------
# Tradeoff curves across parameter sweeps

def plot_tradeoff_sweep(ax, result_dict, title, show_legend=False, show_ylabel=False):
    for label, results in result_dict.items():
        mismatch = np.asarray(results["tradeoff_mismatch"])
        delay = np.maximum(np.asarray(results["tradeoff_delay"]), 0)

        ax.plot(
            mismatch,
            delay,
            marker="o",
            linestyle="None",   # ← FIX
            markersize=4,
            label=label
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.1,
        alpha=0.55,
        label="0.5 ms threshold"
    )

    ax.set_xlabel("Intercompartment spike mismatch (mV)")

    ax.set_title(title)
    ax.title.set_x(0.0)

    ax.grid(True, alpha=0.30)
    ax.set_ylim(bottom=0)

    if show_ylabel:
        ax.set_ylabel("Propagation delay (ms)")
    else:
        ax.set_ylabel("")

    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="upper left")


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

plot_tradeoff_sweep(
    axes[0],
    coupling_results,
    "A. Coupling strength",
    show_legend=True,
    show_ylabel=True
)

plot_tradeoff_sweep(
    axes[1],
    sodium_results,
    "B. Total sodium conductance",
    show_legend=False,
    show_ylabel=False
)

plot_tradeoff_sweep(
    axes[2],
    input_results,
    "C. Input strength",
    show_legend=False,
    show_ylabel=False
)

plt.tight_layout(w_pad=2.2)

plt.savefig("FigS1_parameter_sweep_tradeoff_curves.png", dpi=300, bbox_inches="tight")
plt.savefig("FigS1_parameter_sweep_tradeoff_curves.svg", bbox_inches="tight")

plt.show()


# In[70]:


# ---------- Supplementary Figure S1 ----------
# Pareto-style tradeoff visualization across parameter sweeps

import numpy as np
import matplotlib.pyplot as plt


def get_pareto_frontier(mismatch, delay):
    """
    Identify Pareto-efficient points when both objectives are minimized:
    lower mismatch and lower delay are both better.
    """
    mismatch = np.asarray(mismatch)
    delay = np.asarray(delay)

    pareto = np.ones(len(mismatch), dtype=bool)

    for i in range(len(mismatch)):
        dominated_by_other = (
            (mismatch <= mismatch[i]) &
            (delay <= delay[i]) &
            ((mismatch < mismatch[i]) | (delay < delay[i]))
        )
        if np.any(dominated_by_other):
            pareto[i] = False

    return pareto


def remove_duplicate_mismatch_values(pareto_mismatch, pareto_delay):
    """
    Remove duplicate mismatch values from the Pareto frontier.
    If duplicate mismatch values occur, keep the point with the lowest delay.
    """
    unique = {}

    for m, d in zip(pareto_mismatch, pareto_delay):
        m_key = np.round(m, 10)
        if m_key not in unique or d < unique[m_key]:
            unique[m_key] = d

    cleaned_mismatch = np.array(sorted(unique.keys()))
    cleaned_delay = np.array([unique[m] for m in cleaned_mismatch])

    return cleaned_mismatch, cleaned_delay


def plot_pareto_tradeoff_sweep(
    ax,
    result_dict,
    title,
    legend_labels=None,
    show_ylabel=False
):
    all_mismatch = []
    all_delay = []

    if legend_labels is None:
        legend_labels = list(result_dict.keys())

    for (original_label, results), legend_label in zip(result_dict.items(), legend_labels):
        mismatch = np.asarray(results["tradeoff_mismatch"])
        delay = np.maximum(np.asarray(results["tradeoff_delay"]), 0)

        all_mismatch.append(mismatch)
        all_delay.append(delay)

        ax.scatter(
            mismatch,
            delay,
            s=24,
            alpha=0.45,
            label=legend_label
        )

    all_mismatch = np.concatenate(all_mismatch)
    all_delay = np.concatenate(all_delay)

    pareto_mask = get_pareto_frontier(all_mismatch, all_delay)

    pareto_mismatch = all_mismatch[pareto_mask]
    pareto_delay = all_delay[pareto_mask]

    sort_idx = np.argsort(pareto_mismatch)
    pareto_mismatch = pareto_mismatch[sort_idx]
    pareto_delay = pareto_delay[sort_idx]

    pareto_mismatch, pareto_delay = remove_duplicate_mismatch_values(
        pareto_mismatch,
        pareto_delay
    )

    # Draw the Pareto-like frontier.
    # If duplicate/near-duplicate mismatch values create a vertical artifact,
    # show frontier points without forcing a misleading connecting segment.
    is_strictly_increasing = np.all(np.diff(pareto_mismatch) > 1e-8)

    if is_strictly_increasing:
        ax.plot(
            pareto_mismatch,
            pareto_delay,
            color="black",
            linewidth=1.7,
            alpha=0.80,
            label="Pareto-like frontier"
        )

        ax.scatter(
            pareto_mismatch,
            pareto_delay,
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9
        )
    else:
        ax.scatter(
            pareto_mismatch,
            pareto_delay,
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            label="Pareto-like frontier"
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1.1,
        alpha=0.55,
        label="0.5 ms threshold"
    )

    ax.set_xlabel("Intercompartment Spike Mismatch (mV)")
    ax.set_title(title, loc="left", x=0.0, ha="left")
    ax.grid(True, alpha=0.30)
    ax.set_ylim(bottom=0)

    if show_ylabel:
        ax.set_ylabel("Propagation delay (ms)")
    else:
        ax.set_ylabel("")


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

generic_legend_labels = [
    "Low parameter",
    "Baseline",
    "High parameter"
]

plot_pareto_tradeoff_sweep(
    axes[0],
    coupling_results,
    "A. Coupling Strength",
    legend_labels=generic_legend_labels,
    show_ylabel=True
)

plot_pareto_tradeoff_sweep(
    axes[1],
    sodium_results,
    "B. Total Sodium Conductance",
    legend_labels=generic_legend_labels,
    show_ylabel=True
)

plot_pareto_tradeoff_sweep(
    axes[2],
    input_results,
    "C. Input Strength",
    legend_labels=generic_legend_labels,
    show_ylabel=True
)

# Remove duplicate ylabel text while preserving subplot spacing
axes[1].set_ylabel("")
axes[2].set_ylabel("")

# Figure-level legend, placed outside Panel C
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.94, 0.5),
    frameon=False,
    fontsize=9
)

# Equal spacing + room for legend
plt.subplots_adjust(left=0.08, right=0.92, wspace=0.25)

plt.savefig("FigS1_parameter_sweep_pareto_tradeoff.png", dpi=300, bbox_inches="tight")
plt.savefig("FigS1_parameter_sweep_pareto_tradeoff.svg", bbox_inches="tight")

plt.show()


# In[71]:


# ---------- Supplementary Figure S2 ----------
# Time-step validation

dt_conditions = {
    "Δt = 0.005 ms": 0.005,
    "Δt = 0.01 ms": 0.01,
    "Δt = 0.02 ms": 0.02
}

dt_results = {}

for label, dt_value in dt_conditions.items():
    cluster_success = []
    diffuse_success = []
    mismatch_values = []
    delay_values = []
    valid_cluster_values = []
    feasible_cluster_values = []
    both_spike_indices = []

    cluster_max = []
    diffuse_max = []

    for i, f in enumerate(cluster_values):
        time, V_c, V_d = run_two_compartment_simulation(
            cluster_fraction=f,
            g_c=1.0,
            g_Na_total=120.0,
            I_ext_amp=20.0,
            dt=dt_value
        )

        c_spike = int(check_spike(V_c))
        d_spike = int(check_spike(V_d))

        cluster_success.append(c_spike)
        diffuse_success.append(d_spike)

        Vc_max = max(V_c)
        Vd_max = max(V_d)

        cluster_max.append(Vc_max)
        diffuse_max.append(Vd_max)

        t_c = first_spike_time(time, V_c, threshold=0)
        t_d = first_spike_time(time, V_d, threshold=0)

        if t_c is not None and t_d is not None:
            delay = t_d - t_c
            mismatch = abs(Vc_max - Vd_max)

            # Keep only forward-propagating delay values
            if delay >= 0:
                valid_cluster_values.append(f)
                delay_values.append(delay)
                mismatch_values.append(mismatch)
                both_spike_indices.append(i)

                if delay < delay_threshold:
                    feasible_cluster_values.append(f)

    if len(mismatch_values) > 0:
        best_local_index = mismatch_values.index(min(mismatch_values))
        optimal_cluster_fraction = cluster_values[both_spike_indices[best_local_index]]
    else:
        optimal_cluster_fraction = None

    if len(feasible_cluster_values) > 0:
        feasible_min = min(feasible_cluster_values)
        feasible_max = max(feasible_cluster_values)
    else:
        feasible_min = None
        feasible_max = None

    dt_results[label] = {
        "valid_cluster_values": valid_cluster_values,
        "mismatch_values": mismatch_values,
        "delay_values": delay_values,
        "optimal_cluster_fraction": optimal_cluster_fraction,
        "feasible_min": feasible_min,
        "feasible_max": feasible_max
    }


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for label, results in dt_results.items():
    axes[0].plot(
        results["valid_cluster_values"],
        results["mismatch_values"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label=label
    )

    axes[1].plot(
        results["valid_cluster_values"],
        results["delay_values"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label=label
    )

axes[0].set_title("A. Mismatch", loc="left", x=0.0, ha="left")
axes[0].set_xlabel("Cluster Fraction")
axes[0].set_ylabel("Intercompartment Mismatch (mV)")
axes[0].grid(True, alpha=0.30)

axes[1].set_title("B. Propagation Delay", loc="left", x=0.0, ha="left")
axes[1].set_xlabel("Cluster Fraction")
axes[1].set_ylabel("Propagation Delay (ms)")
axes[1].axhline(0.5, linestyle="--", linewidth=1.1, alpha=0.55)
axes[1].grid(True, alpha=0.30)
axes[1].set_ylim(bottom=0)

axes[1].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 0.86, 1])

plt.savefig("FigS2_time_step_validation.png", dpi=300, bbox_inches="tight")
plt.savefig("FigS2_time_step_validation.svg", bbox_inches="tight")

plt.show()


# In[72]:


# ---------- Supplementary Figure S3 ----------
# Representative voltage traces for the three regimes

representative_conditions = {
    "A. Low Clustering | Stimulus-Site Initiation Failure": 0.20,
    "B. Intermediate Clustering | Dual Spiking": 0.50,
    "C. High Clustering | Propagation Failure": 0.85
}

fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

for ax, (title, cluster_fraction) in zip(axes, representative_conditions.items()):
    time, V_c, V_d = run_two_compartment_simulation(
        cluster_fraction=cluster_fraction,
        g_c=1.0,
        g_Na_total=120.0,
        I_ext_amp=20.0,
        dt=0.01
    )

    ax.plot(
        time,
        V_c,
        linewidth=1.3,
        label="Clustered Compartment"
    )

    ax.plot(
        time,
        V_d,
        linewidth=1.3,
        label="Diffuse Compartment"
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.3,
        alpha=0.25
    )

    ax.set_title(
        f"{title} (Cluster Fraction = {cluster_fraction:.2f})",
        loc="left",
        x=0.0,
        ha="left"
    )

    ax.set_ylabel("Voltage (mV)")
    ax.grid(True, alpha=0.30)

axes[-1].set_xlabel("Time (ms)")


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.88, 0.5),
    frameon=False,
    fontsize=9
)


plt.tight_layout(rect=[0, 0, 0.9, 1], h_pad=2.0)

plt.savefig("FigS3_representative_voltage_traces.png", dpi=300, bbox_inches="tight")

plt.show()


# In[73]:


cluster_values_full = np.linspace(0, 1, 200)


# In[74]:


results = []

for f in cluster_values_full:
    time, V_c, V_d = run_two_compartment_simulation(cluster_fraction=f)

    cluster_spike = check_spike(V_c)
    diffuse_spike = check_spike(V_d)

    Vc_max = max(V_c)
    Vd_max = max(V_d)

    t_cluster = first_spike_time(time, V_c)
    t_diffuse = first_spike_time(time, V_d)

    delay = None
    if t_cluster is not None and t_diffuse is not None:
        delay = t_diffuse - t_cluster

    mismatch = abs(Vc_max - Vd_max)

    results.append({
        "f": f,
        "cluster_spike": cluster_spike,
        "diffuse_spike": diffuse_spike,
        "Vc_max": Vc_max,
        "Vd_max": Vd_max,
        "delay": delay,
        "mismatch": mismatch
    })


# In[76]:


f_vals = np.array([r["f"] for r in results])
cluster_spike = np.array([r["cluster_spike"] for r in results])
diffuse_spike = np.array([r["diffuse_spike"] for r in results])
delay_vals = np.array([r["delay"] for r in results])
mismatch_vals = np.array([r["mismatch"] for r in results])

# Initiation failure
init_fail = f_vals[cluster_spike == False]

# Dual-spiking region
dual_spike = f_vals[(cluster_spike == True) & (diffuse_spike == True)]

# Feasible region
feasible = f_vals[
    (cluster_spike == True) &
    (diffuse_spike == True) &
    (np.array([d if d is not None else np.nan for d in delay_vals]) < 0.5)
]

# Valid dual-spiking points
valid = (cluster_spike == True) & (diffuse_spike == True)
valid_f = f_vals[valid]
valid_mismatch = mismatch_vals[valid]
valid_delay = delay_vals[valid]

# Mismatch optimum
idx_mismatch = np.argmin(valid_mismatch)
f_mismatch_opt = valid_f[idx_mismatch]
min_mismatch = valid_mismatch[idx_mismatch]
delay_at_mismatch = valid_delay[idx_mismatch]

# Delay optimum
idx_delay = np.nanargmin(valid_delay)
f_delay_opt = valid_f[idx_delay]
min_delay = valid_delay[idx_delay]

print("Initiation failure range:", init_fail.min(), "to", init_fail.max())
print("Dual-spiking range:", dual_spike.min(), "to", dual_spike.max())
print("Feasible range:", feasible.min(), "to", feasible.max())
print("Mismatch-minimizing α:", f_mismatch_opt)
print("Minimum mismatch:", min_mismatch)
print("Delay at mismatch optimum:", delay_at_mismatch)
print("Delay-minimizing α:", f_delay_opt)
print("Minimum delay:", min_delay)


# In[77]:


print(delay_at_mismatch)
print(abs(delay_at_mismatch) < 0.5)


# In[78]:


import numpy as np
from scipy.integrate import solve_ivp

# Hodgkin-Huxley rate functions
def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)) if abs(V + 40) > 1e-9 else 1.0

def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)) if abs(V + 55) > 1e-9 else 0.1

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


def rk45_two_compartment_simulation(
    cluster_fraction,
    gNa_total=120.0,
    gK=36.0,
    gL=0.3,
    ENa=50.0,
    EK=-77.0,
    EL=-54.4,
    Cm=1.0,
    gc=1.0,
    I0=20.0,
    t_start=0.0,
    t_end=50.0,
    dt_output=0.01
):
    gNa_c = cluster_fraction * gNa_total
    gNa_d = (1 - cluster_fraction) * gNa_total

    V0 = -65.0

    m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
    h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
    n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))

    # state = [Vc, mc, hc, nc, Vd, md, hd, nd]
    y0 = [V0, m0, h0, n0, V0, m0, h0, n0]

    def I_stim(t):
        return I0 if 5.0 <= t <= 7.0 else 0.0

    def rhs(t, y):
        Vc, mc, hc, nc, Vd, md, hd, nd = y

        INa_c = gNa_c * (mc ** 3) * hc * (Vc - ENa)
        IK_c = gK * (nc ** 4) * (Vc - EK)
        IL_c = gL * (Vc - EL)
        Icouple_c = gc * (Vd - Vc)

        dVc = (-INa_c - IK_c - IL_c + Icouple_c + I_stim(t)) / Cm
        dmc = alpha_m(Vc) * (1 - mc) - beta_m(Vc) * mc
        dhc = alpha_h(Vc) * (1 - hc) - beta_h(Vc) * hc
        dnc = alpha_n(Vc) * (1 - nc) - beta_n(Vc) * nc

        INa_d = gNa_d * (md ** 3) * hd * (Vd - ENa)
        IK_d = gK * (nd ** 4) * (Vd - EK)
        IL_d = gL * (Vd - EL)
        Icouple_d = gc * (Vc - Vd)

        dVd = (-INa_d - IK_d - IL_d + Icouple_d) / Cm
        dmd = alpha_m(Vd) * (1 - md) - beta_m(Vd) * md
        dhd = alpha_h(Vd) * (1 - hd) - beta_h(Vd) * hd
        dnd = alpha_n(Vd) * (1 - nd) - beta_n(Vd) * nd

        return [dVc, dmc, dhc, dnc, dVd, dmd, dhd, dnd]

    t_eval = np.arange(t_start, t_end + dt_output, dt_output)

    sol = solve_ivp(
        rhs,
        (t_start, t_end),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )

    time = sol.t
    Vc = sol.y[0]
    Vd = sol.y[4]

    return time, Vc, Vd


# In[84]:


def spike_time(time, V, threshold=0.0):
    above = np.where(V >= threshold)[0]
    if len(above) == 0:
        return None
    return time[above[0]]

def spike_present(V, threshold=0.0):
    return np.max(V) >= threshold

def compute_metrics(time, Vc, Vd):
    Vc = np.array(Vc)
    Vd = np.array(Vd)
    time = np.array(time)

    cluster_spike = spike_present(Vc)
    diffuse_spike = spike_present(Vd)

    mismatch = abs(np.max(Vc) - np.max(Vd))

    tc = spike_time(time, Vc)
    td = spike_time(time, Vd)

    delay = None
    if tc is not None and td is not None:
        delay = td - tc

    return {
        "cluster_spike": cluster_spike,
        "diffuse_spike": diffuse_spike,
        "cluster_peak": np.max(Vc),
        "diffuse_peak": np.max(Vd),
        "mismatch": mismatch,
        "delay": delay,
        "delay_magnitude": abs(delay) if delay is not None else None
    }


# In[85]:


test_alphas = [0.30, 0.56, 0.70]

rk45_results = []

for alpha in test_alphas:
    time_rk, Vc_rk, Vd_rk = rk45_two_compartment_simulation(cluster_fraction=alpha)
    metrics = compute_metrics(time_rk, Vc_rk, Vd_rk)

    rk45_results.append({
        "alpha": alpha,
        **metrics
    })

for r in rk45_results:
    print("\nalpha =", r["alpha"])
    print("cluster spike:", r["cluster_spike"])
    print("diffuse spike:", r["diffuse_spike"])
    print("cluster peak:", r["cluster_peak"])
    print("diffuse peak:", r["diffuse_peak"])
    print("mismatch:", r["mismatch"])
    print("delay:", r["delay"])
    print("delay magnitude:", r["delay_magnitude"])


# In[86]:


run_two_compartment_simulation(cluster_fraction=alpha)


# In[87]:


comparison_results = []

for alpha in test_alphas:
    # Euler
    time_eu, Vc_eu, Vd_eu = run_two_compartment_simulation(cluster_fraction=alpha)
    euler_metrics = compute_metrics(time_eu, Vc_eu, Vd_eu)

    # RK45
    time_rk, Vc_rk, Vd_rk = rk45_two_compartment_simulation(cluster_fraction=alpha)
    rk45_metrics = compute_metrics(time_rk, Vc_rk, Vd_rk)

    comparison_results.append({
        "alpha": alpha,
        "Euler mismatch": euler_metrics["mismatch"],
        "RK45 mismatch": rk45_metrics["mismatch"],
        "Euler delay magnitude": euler_metrics["delay_magnitude"],
        "RK45 delay magnitude": rk45_metrics["delay_magnitude"]
    })

for r in comparison_results:
    print("\nalpha =", r["alpha"])
    print("Euler mismatch:", r["Euler mismatch"])
    print("RK45 mismatch:", r["RK45 mismatch"])
    print("Euler delay magnitude:", r["Euler delay magnitude"])
    print("RK45 delay magnitude:", r["RK45 delay magnitude"])


# In[ ]:




