# This program can create and save the images of graphs used in the dataset. 

path = 'C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/'
for i in range(5,16): # number of vertices 
    for j in range (i+(int(i/2)), i*(i-1)/2): # number of edges
        for k in range(10): # number of copies of each graph with i vertices and j edges
            g = graphs.RandomGNM(i, j)
            name = path + str(i) + 'x' + str(j) + 'x' + str(k) + '.png'
            g.plot().save(name)
