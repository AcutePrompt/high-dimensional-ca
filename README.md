# Self-Domestication in Complex Systems: Emergent Simplicity in a Self-Modifying Hypergraph Network  

---

## Abstract  
We present a computational study of a self-modifying hypergraph network where nodes dynamically rewire connections, mutate update rules, and interact via a global diffusive field. Despite built-in capacities for chaos (nonlinear activations, stochastic mutations), the system consistently collapses into two stable clusters—a high-coherence "core" and a passive "background." By testing architectural variants (activation functions, rewiring, diffusion rates), we demonstrate that this simplicity is *emergent*, not hardcoded. We link this behavior to universal principles in complex systems, with parallels in biology, sociology, and machine learning.  

---

## 1. Introduction  
**Self-modifying systems**—from evolving ecosystems to adaptive neural networks—are theorized to exhibit perpetual novelty. However, our simulation reveals a counterintuitive phenomenon: *self-domestication*, where freedom to mutate/rewire leads not to chaos but to stable simplicity.  

**Real-World Example**:  
*Social media algorithms (which evolve based on user engagement) often homogenize content into echo chambers, despite their capacity for diversity—a "self-domesticating" outcome similar to our clusters.*  

---

## 2. Methodology  
### 2.1 Simulation Design  
- **Nodes**: \( N = 50 \times 50 \) grid, \( D \)-dimensional states (\( D = 2 \)–\( 256 \)).  
- **Dynamics**: Asynchronous updates, stochastic mutations (\( \mu_{\text{param}} = 0.01 \)), curiosity-driven rewiring (\( \delta_{\text{thresh}} = 1.0 \)).  
- **Global Field**: Diffusive coupling (\( \alpha = 0.1 \)–\( 0.9 \)) with random pulses.  

### 2.2 Tested Variants  
- Disabled rewiring, sigmoid-only activation, varied \( \alpha \).  

**Real-World Example**:  
*Ecological networks, where species interactions (rewiring) and trait evolution (mutations) are constrained by global factors like climate (\( F \)).*  

---

## 3. Results  
### 3.1 Emergent Bifurcation  
- **Two-Cluster Attractor**: 75% background (near-zero states) and 25% core (high-amplitude, \( \sim 0.5 \)) emerge across all dimensions (Fig. 1A).  
- **Silhouette Scores**: \( \sim 0.78 \), indicating sharp separation.  
- **Predictability**: Core cluster dynamics become linearly explainable (\( R^2 \uparrow \) with dimensionality, peaking at 0.873 in 256D).  

**Real-World Example**:  
*Traffic flow phase transitions—free flow (Cluster 0) vs. synchronized congestion (Cluster 1)—emerge from local driver rules, not centralized control.*  

### 3.2 Nonlinearity as a Catalyst  
- **Sigmoid Activation**: Destroys clusters (silhouette = 0.022) but maximizes predictability (\( R^2 = 0.971 \)).  
- **Nonlinear Activations (sine/relu)**: Enable bifurcation by allowing *asymmetric saturation*.  

**Real-World Example**:  
*Neural activity in the brain—balanced excitation (relu-like) and inhibition (sigmoid-like)—prevents epileptic seizures (homogenization) while enabling complex computation.*  

### 3.3 Dimensionality Stabilizes Order  
- **Higher Dimensions**: Increase core predictability (\( R^2 \) from 0.447 in 2D to 0.873 in 256D).  

**Real-World Example**:  
*High-dimensional financial markets (e.g., derivatives) stabilize through diversification, while low-dimensional markets (e.g., cryptocurrencies) remain volatile.*  

---

## 4. Discussion  
### 4.1 Self-Domestication as a Universal Attractor  
Our system’s collapse into simplicity mirrors:  
- **Biological Senescence**: Aging organisms lose plasticity despite genomic "freedom."  
- **Cultural Homogenization**: Societies converge on dominant languages/traditions.  

**Real-World Example**:  
*Wikipedia’s edit wars—a self-modifying system—ultimately stabilize into consensus articles.*  

### 4.2 Challenging the Edge of Chaos Paradigm  
- **Stability Begets Stability**: Coherence suppresses exploration.  
- **Mutation ≠ Innovation**: Parametric noise reinforces order.  

**Real-World Example**:  
*Corporate bureaucracies stagnate despite incentives for innovation.*  

---

## 5. Implications  
### 5.1 For Artificial Life  
- **Open-Ended Evolution**: Requires decentralized control and asymmetric rewards.  
- **Warning**: Risk of "premature convergence."  

**Real-World Example**:  
*AlphaFold’s protein-folding breakthroughs required constrained search spaces.*  

### 5.2 For Machine Learning  
- **Overparameterization**: High-dimensional models may collapse into too much order.  
- **Recommendation**: Inject "chaos preservation" mechanisms.  

**Real-World Example**:  
*GPT-4’s "dulling" over training—chaotic creativity tamed into predictability.*  

---

## 6. Conclusion  
We identify *self-domestication* as a fundamental behavior of complex systems with:  
1. **Nonlinear local rules**,  
2. **Global feedback** (\( F \)),  
3. **Parametric flexibility** (mutations/rewiring).  

---

## 7. Future Directions  
1. Test biological analogs (e.g., gene regulatory networks).  
2. Explore interventions to sustain chaos (e.g., time-varying \( \alpha \)).  
3. Map to real-world data (e.g., social media echo chambers).  

**Real-World Example**:  
*Urban planning—could cities avoid self-domestication (homogeneity) while retaining coherence?*  

