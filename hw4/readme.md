# HW4

## backpropagation
### neuralNetwork.py
- Layer class
    - create a layer with n neuron by pandas dataframe
    - each columns refer to a neuron with it weight and bias
    - when initial a layer, will generate weight randomly

- Network class
    - call genNetwork() function to generate a neuron network 
        - 2 hidden layer & 5 neuron each hidden layer by default
    - store every layer by list
    - save/load the network by pickle package