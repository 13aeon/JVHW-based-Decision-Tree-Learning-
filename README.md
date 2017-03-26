# decision-tree
A JVHW estimator based Decision Tree python implementation with validation, pruning, and attribute multi-splitting
Contributors: Banghua Zhu

Note:
The original version of decision tree we rely on is from https://github.com/ryanmadden/decision-tree, the author: Ryan Madden and Ally Cody

## Requirements

python 2.7.6 [Download](https://www.python.org/download/releases/2.7.6/)

numpy, scipy and matlab module in python

## Files
* .train, .test file - The training, and testing sets used for building and testing the program (It also supports validation but we didn't include in our test.)
* decision-tree.py - The decision tree program
* datatypes.txt - A metadata file that indicates (with comma separated true/false entries) which attributes are numeric (true) and nominal (false) **Note: You must edit this file or supply your own if using a different dataset than the one provided**

## How to run
decision-tree.py accepts parameters passed via the command line. The possible paramters are:
* Filename for training (Required, must be the first argument after 'python decision-tree.py')
* Estimator (-e) followed by the name of estimator to use (Required, the name should be either JVHW or MLE)
* Method (-m) followed by the name of method to use (Required, the name could be C4.5, ID3 or CART)
* Classifier name (Optional, by default the classifier is the last column of the dataset)
* Datatype flag (-d) followed by datatype filename (Optional, defaults to 'datatypes.csv')
* Print flag (-s) (Optional, causes the dataset)
* Validate flag (-v) followed by validate filename (Optional, specifies file to use for validation)
* Test flag (-t) followed by test filename (Optional, specifies file to use for testing)
* Pruning flag (-p) (Optional, you must include a validation file in order to prune)

#### Examples
```
python decision-tree.py data/cmc.train -d data/cmctype.txt -t data/cmc.test -m CART -e MLE
```
This command runs MLE based CART with cmc.train as the training set, cmc.test as the test set, cmctype.txt as the datatype file and pruning disabled. The classifier is not specified so it defaults to the last column in the training set. Printing is not enabled. (In order to view the accuracy, we recommend to use -v instead of -t. The validation is equal to test under this circumstance.)