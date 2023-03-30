# Model for predicting toxicity in comments
This model is a neural network designed to classify comments based on their level of toxicity. It takes a comment as input and predicts the likelihood of it being toxic, severely toxic, obscene, threatening, insulting, or containing identity hate. The model is implemented using the Keras library in Python.

# Model Architecture
The model consists of four layers:

## Embedding layer: 
This layer takes the input, which is a comment, and converts it into a fixed-length vector representation. It uses the Embedding function from Keras to map each word to a dense vector of fixed size.

## Bidirectional LSTM layer: 
This layer processes the input sequence of word vectors in both forward and backward directions to capture the context of the comment. It contains 32 LSTM units with a hyperbolic tangent activation function.

## Feature extractor fully connected layers: 
These are three dense layers with 128, 256, and 128 units respectively. They use the Rectified Linear Unit (ReLU) activation function to extract relevant features from the output of the LSTM layer.

## Final layer: 
This is a dense layer with six units and a sigmoid activation function. It predicts the probability of each of the six classes: toxic, severely toxic, obscene, threatening, insulting, and identity hate.

# Input and Output
The model takes a comment as input, which is a string of text. The comment is preprocessed to remove any unwanted characters and converted into a sequence of word vectors using the tokenizer. The output of the model is a vector of six probabilities, one for each class, indicating the likelihood of the comment belonging to each class.

# Training
The model is trained on a dataset of labeled comments using binary cross-entropy loss and the Adam optimizer. The training data is split into training and validation sets, and early stopping is used to prevent overfitting.

# Usage
To use the model, first, load the weights of the pre-trained model. Then, preprocess the comment text and convert it into a sequence of word vectors using the same tokenizer that was used during training. Finally, pass the sequence to the model's predict method to obtain the probabilities of the comment belonging to each class.

# Evaluation
The model's performance is evaluated using several metrics, including accuracy, precision, recall, and F1 score. The evaluation is done on a test dataset that is independent of the training and validation sets.

# Gradio App hosted using .ipynb file and it's screenshot
[Screenshot](https://www.flickr.com/photos/198013377@N04/52780658642/)

# Conclusion
This model can be used to automatically detect and flag potentially toxic comments in online forums, social media, or other online platforms. It can help to prevent online harassment and hate speech and make online communities safer and more inclusive.
