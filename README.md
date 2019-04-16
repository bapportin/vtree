# vtree
Index for multidimensional data for nearest neighbour search
the basic idea:
- split a node by randomly choosen points into subnodes
- attach each node to the subnode with the smallest distance
- for knn search maintain tau (the search radius) distance from query point to the farthest point of the currently found n nearest. if not n have been found, use infinity
- start with the closest node to the query point
- only need to look into nodes which are no futher than (the closest node+tau) away from query point
