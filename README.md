# Decision-Tree-Guided-Deep-Learning

Decision Trees distribute computations by assigning data independently to the left / right child of each node. The recursive application then yields an overall assignment of data to independent leaves (=d-dimensional hyperboxes). In each leave node, we use the mean (for regression) or the majority class (for classification) as a prediction. This makes DTs nice from a computational point of view, but their performance often lacks compared to other methods. 

In this project, we want to combine DTs with Deep Learning by training independent networks on the data in each leaf. This way, we can hopefully use tiny networks in each leaf node while retaining overall performance due to their specialization. The goal is to have a smaller model and/or faster model application in the end without loss in accuracy.
