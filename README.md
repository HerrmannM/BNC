# MonashML
Machine Learning Project at Monash University, Melbourne, Australia

## Compiling and Launching
After cloning the git repository, use `ant` to compile the classes and create a jar.
```
ant
```
The resulting jar can be found in the `dist` folder.
It embeds the necessary dependencies, so using the software is simply a matter of calling java with the `-jar` flag.
For example:
```
java -jar dist/MonashML.jar --model ESKDB -E 5 -I 1000 -K 5 -L 2 --evaluator holdout 0.5 /path/to/file.arff
```

The `--model` and `--evaluator` flags are required.

### Model examples
* Selective KDB: `--model SKDB -I 1000 -K 5 -L 2`
* Ensemble Selective KDB `--model ESKDB -E5 -I 1000 -K 5 -L 2`

Parameters:
* `-I`: Gibbs sampling iteration
* `-K`: The number of parents
* `-L`: Tying on LEVEL, the nodes on the same LEVEL share the same concentration parameter
* `-E`: Ensemble size of ESKDB

### Evaluator examples
* Holdout: `--evaluator holdout 0.5 /path/to/file.arff`
  * Parameters: the ratio of the dataset in the "test" and a dataset arff file
* K-Fold Cross Validation `--evaluator kfoldxval 2 5 /path/to/file.arff`
  * Parameters: The number of splits, the number of rounds and a dataset arff file

### Dependencies
* [Apache Common Maths 3 - 3.6.1](https://commons.apache.org/proper/commons-math/)
* [Apache Common CLI - 1.4](https://commons.apache.org/proper/commons-cli/)
* [Weka - 3.8.3](https://www.cs.waikato.ac.nz/ml/weka/index.html)
* [MonashMLHDP](https://github.com/HerrmannM/MonashMLHDP)
* [MonashMLTools](https://github.com/HerrmannM/MonashMLTools)

## Contributors
* [Dr. François Petitjean](https://github.com/fpetitjean)
* [He Zhang](https://github.com/icesky0125)
* [Dr. Matthieu Herrmann](https://github.com/HerrmannM)
* [Dr. Nayyar Zaidi](https://github.com/nayyarzaidi)
