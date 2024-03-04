
## Activation function
Activation functions are used to have a granular control over the activation of certain neurons.
Activation function makes it so that the output of the network fits the scattered input data

![[why_activation_function.mkv]]



## Softmax Activation
- Negative values = problem
- Don't want negative values.
- Can get rid of negative using ReLU but meaning of negatives is lost
- Welcome exponentitaion
- Negative values converted into positive while still retaining some meaning
- because exponentiation values go biiiiiiiiiiiiiiiiiiiiiiiiiiig
- we "normatlize" them using probablity distribution

Steps:
$$
\text{1. For a neuron output z,   }
z = e^z
$$
$$
\text{2. For all such z, apply probablity distribution,   }
\large\frac{z_i}{\sum_{i=1}^{n}{z_i}}
$$
**Softmax Function:**
$$
\Large
S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l=1}^{L}{e^{z_{i,j}}}}
$$
$l = \text{layer}$
$z = \text{neuron output}$


- Also because exponentiation = big value, Python says no to biiig values.
- So, subtract maximum of 1 input layer to each in the input layer
- Then do the exponentiation
- This guarantees and output between 0 and 1
- After probability distribution, the final output is same as that when not doing the subtraction