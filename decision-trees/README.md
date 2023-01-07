Participants:
 *      Alexis Baladón
 *      Ignacio Viscardi
 *      Rafael Castelli

============================================================================================================

### Main programs:

- ID3.py, provides the necessary functionalities to generate the ID3 tree whith the implemented algorithm. 

- tree.py, provides the necessary functionalities to predict the result of a given tuple. 

- main.py, which lets you execute different algorithms as well as use different datasets in order to evaluate their results (Each flag is properly documented in main.py). 

Example:

To run the program with an unbalanced dataset and using our ID3 implementation (default mode), you should run:

```
py .\main.py
```

To run the program with less unbalanced datasets you can run:

```
py .\main.py --random_oversampling 1 --unbalanced_dataset 0
```

or

```
py .\main.py --smotenc_oversampling 1 --unbalanced_dataset 0
```

or

```
py .\main.py --undersampling 1 --unbalanced_dataset 0
```

The sklearn_tree flag can be used to execute skelarn's ID3 implementation. Furthermore, sklearn_tree's algorithm variation can also be changed using the flag --sklearn_tree_type <type> (Default: "entropy"):

```
python main.py --sklearn_tree 1 --sklearn_tree_type "gini"
```
