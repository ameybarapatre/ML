# ML
Matelabs Problem is classify 5 Different Shapes
Viz. Triangles , Circle , Rectangle , Pentagon , Hexagon



Shapes.zip has train , test and validation data
images are of dimensions 28 X 28

Data.zip has the single channel flattened image tensors of the shapes data

Convolutional Model:

It has 4 CNN layers ,2 Norm layers , 2 Max Pool layers , the final layers are two densely connected layers
with 5 outputs as logits.

Convolutional Model yeilds 0.47 Accuracy


SVM Classifier :

PCA is applied and dimensionality is reduced to 50 ,from 784 (28 x 28)




