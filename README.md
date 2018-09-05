# NNValidated.py
In order to run the NNValidated.py one must put the training files into the Data/Train and the test file into Data/Test. The prediction file is located at env/ directory.
After this one must only call NNValidate.py that the script will train, using GridSearchCV, the best NeuralNet with the hyperparameter defined.

# Description
I have chosen a one Hidden Layer Neural Net because most of the problems, as stated on the literature, one single layer is enough for approximation. So I have chosen a simple approach.

I Have used a Root Mean Squared Error as the cost function because I can train both outputs on the same magnitude. This cost function is the same magnitude level of Mean Absolute Error and do not diverge from Mean Squared Error. So I decided that would be a good fit since we are checking the outputs with different metrics.

The dropout layers are in order to avoid overfitting.
