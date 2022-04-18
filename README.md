# ML-GraphVertexEdgeCounter
A machine learning project in PyTorch with the aim to interpret the image of a graph and return its numbers of vertices. 

This was an individual project for class, namely CS320: Software Engineering and Design at York College of Pennsylvania, Spring 2022. 

The neural network does not achieve beyond 20% accuracy. It appears that the training gets stuck between loss values of 2.67 to 2.72. I speculate the issue is with the graph image data. The dimensions of the images vary significantly. Due to rescaling the images for the neural network, the shape of the vertices is not consistent. Hence, this neural network would likely need more consistently-sized graph data in order to function properly. 

Various tutorials were referenced for this project, which are noted in the comments of each file. 