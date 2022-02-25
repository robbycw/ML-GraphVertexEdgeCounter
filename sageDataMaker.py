# This program can create and save the images of graphs used in the dataset. 

path = 'C:/Users/bsawe/Documents/GitHub/ML-GraphVertexEdgeCounter/graphdata/'
for i in range(3,10): # number of vertices 
    for j in range (i, i*(i-1)/2): # number of edges
        g = graphs.RandomGNM(i, j)
        name = path + str(i) + 'x' + str(j) + '.png'
        g.plot().save(name)
