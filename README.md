# The Structured Weighted Violation MIRA

Code for "The Structured Weighted Violation MIRA" by Dor Ringel, Rotem Dror, and Roi Reichart

## Dependencies
In order to run the program the following dependencies need to be installed:

- Java 1.8:
    - Install Java 1.8 SDK and choose it as the project's SDK.
    
- Gurobi optimization package:
    - Download Gurobi from their website
    - Install it
    - Create and activate a license (free for academic purposes)
    - Add installed gurobi.jar to the project's SDKs
    
- Weka machine learning toolkit:
    - Download Weka from their website
    - Install it
    - Add the installed weka.jar to the project's SDKs
    
- Mallet machine learning toolkit:
    - Install via Maven, using address: <cc.mallet:mallet:2.0.8>
    
- Gson Java to Json converter
    - Install via Maven, using address: <com.google.code.gson:gson:2.8.2>
    
## Running the code

### Endpoint
Running the code is done by the main method in Main class, located in: src/main/java/examples/Main.java

The parameters can be passed into the various experiments using the cvParams object (defined in src/main/java/examples/Main.java)

### General parameters
There are three mandatory general parameters:
- dataset: dataset to use. can be any of the following:
    - genia
    - bc2gm
    - conll2002
    - conll2000-chunking
    
- numTrainIterations: number of iteration for the training process.
    - Has to be a positive integer between 1 and 30.
    
- maxFolds: number of folds to run.
    - Has to be a positive integer between 1 and 9.

### Algorithm specific parameters
Additional algorithm-specific parameters include:
- UpdatorClass - which updator algorithm to use.

- kMira: K for MIRA (when choosing updatorClass KBestMiraUpdator).
    - Has to be a positive integer between 1 and 9.

- topK: top K inference results (when choosing updatorClass SWVPUpdator or SWVMUpdaor).
    - Has to be a positive integer between 1 and 9.
    
- maType: type of modification templates to use (when choosing updatorClass SWVPUpdator or SWVMUpdaor).
    - can be aggresive, passive, or all.
    
- jjTypw: size of modification templates to use (when choosing updatorClass SWVPUpdator or SWVMUpdaor).
    - can be single or double.
    
- gammaObjectiveType: type of objective function (when choosing updatorClass SWVPUpdator or SWVMUpdaor, and gammaCalculationMethod four).
    - can be MAXIMIZE or MINIMIZE.
    
- gammaCalculationMethod: method for selecting gammas (when choosing updatorClass SWVPUpdator or SWVMUpdaor).
    - can be uniform, wm, wmr, softmin or four.
    
- gammaCalculationBeta: beta value for gamma calculation (when choosing updatorClass SWVPUpdator or SWVMUpdaor).
    - can be a positive decimal number.
  
  
## Logging and output files
Each run creates a output-directory in src/main/java/examples/output_files/cv/{dataset}, where {dataset} refers to the chosen dataset. This output-directory contains average, fold-level, and sentence-level statistics and also the raw log file.

Logging is also sent to stdout.


## Examples

### Example #1 - CSP:
In order to run CSP algorithm on the BC2GM dataset for 10 iteration and 2 on folds the following parameters should be specified:

cvParams.put("dataset", "bc2gm"); \
cvParams.put("numTrainIterations", "10"); \
cvParams.put("maxFolds", "2"); \
cvParams.put("updatorClass", "PerceptronUpdator");

### Example #2 - SWVM:
In order to run SWVM algorithm on the CONLL2002 dataset for 15 iteration and 5 on folds, using size one modification templates, updating in respect to top three inference results, aggressive approach, and uniform gamma selection, the following parameters should be specified:

gammaObjectiveType and gammaCalculationBeta parameters will be ignored (due to the selection of uniform approach) but a legal value has to be specific for both of them regardless. 

cvParams.put("dataset", "conll2002"); \
cvParams.put("numTrainIterations", "15"); \
cvParams.put("maxFolds", "5"); \
cvParams.put("updatorClass", "SWVMUpdator"); \
cvParams.put("topK", "3");                          \
cvParams.put("maType", "aggresive");                      \
cvParams.put("jjType", "single");                   \
cvParams.put("gammaCalculationMethod", "uniform");     

cvParams.put("gammaObjectiveType", "MAXIMIZE");     \
cvParams.put("gammaCalculationBeta", "1");  


## Citation

### The Structured Weighted Violation MIRA

[1] D. Ringel, R. Dror, R. Reichart [*The Structured Weighted Violation MIRA*](https://arxiv.org/abs/2005.04418)

```
@misc{ringel2020structured,
    title={The Structured Weighted Violations MIRA},
    author={Dor Ringel and Rotem Dror and Roi Reichart},
    year={2020},
    eprint={2005.04418},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


Contact: [dorringel@cs.technion.ac.il](mailto:dorringel@cs.technion.ac.il)