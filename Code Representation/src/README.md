## Details of Files in ./main/java

### ast
- **AST**:  Generate AST dot file for each method and store save them in dir. mentioned in the class.\
- **ASTEdge & ASTNode**:  Classes needed in AST.class
### call
- **Call**:  Determining the caller-callee relationship of methods
### cfg
- **CFG**: Generate CFG of methods
- **CFGEdge, CFGGraph, CFGNode**: Classes needed in CFG.class
###cli
- **CLI**: Command Line and Text Embedding program
###compile
- **Compile2Jar**: Convert Java Class to Jar format
###config
- **ALL**: Path and graph configuration for Java classes
###detection
- **Detection**: Clone Detection using Word2vec and HOPE for evaluation
###embedding
- **Graph2vec**: Apply Graph2vec and output embeddings
- **HOPE**: Apply HOPE and output embeddings
- **Word2vec**: Apply Word2vec and output embeddings
###feature
- **Feature**: Get CFG of method and merge them into new graph following caller-callee relationship
###function
- **Function**:Class containing helper functions
###joern
- **CPG**: Code Property Graph manipulation
###method
- **Method**: Merge AST and CFG of methods
- Other: Helper function to parse the source file method and extract information
###process
- All Files: Classes to execute other codes like Neural Model, graph2vec, word2vec and so on
###test
- **GenerateTrainingData**: Main file for generating dataset
###tool
- **Tool**: Contains all helper functions needed in GenerateTrainingData