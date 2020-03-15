# great_reads

## Basic Concepts

**Distributions**
* Poisson:
* Binomial: 
* Gaussian:
* Chi-squared:
* Negative Binomial: 

**MLE vs. EM**:
* MLE is for a single distribution (finding parameters that maximize likelihood e.g. regression problems)
* EM is for data coming from multiple distributions - finding which distribution a data point is coming from (clustering)
* https://math.stackexchange.com/questions/1390804/what-is-the-relationship-or-difference-between-mle-and-em-algorithm
* https://www.quora.com/How-can-I-tell-the-difference-between-EM-algorithm-and-MLE
* https://stats.stackexchange.com/questions/201194/confusion-about-the-mle-vs-em-algorithm/201202

**Latent Variables**
* Hidden Variables that are not observed
* https://bloomberg.github.io/foml/#lecture-27-em-algorithm-for-latent-variable-models

**Imputing Missing Values**
* https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779  

**Imbalanced Datasets**
* https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167  
* Cross Validation on Imbalanced Datasets:  https://medium.com/lumiata/cross-validation-for-imbalanced-datasets-9d203ba47e8

**Model Evaluation Metrics (Log Loss)**
* https://becominghuman.ai/understand-classification-performance-metrics-cad56f2da3aa  

**Data Leakage**
* https://towardsdatascience.com/data-leakage-part-i-think-you-have-a-great-machine-learning-model-think-again-ad44921fbf34  

**Dimensionality Reduction Techniques**
* https://towardsdatascience.com/dimension-reduction-techniques-with-python-f36ca7009e5c  
* Visualizing data with t-SNE: https://youtu.be/RJVL80Gg3lA
* t-SNE in detail: https://distill.pub/2016/misread-tsne/

**Feature Engineering**
* https://towardsdatascience.com/why-automated-feature-engineering-will-change-the-way-you-do-machine-learning-5c15bf188b96 
* https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219  

**Feature Selection**
* https://towardsdatascience.com/feature-selection-and-dimensionality-reduction-f488d1a035de 

**Hyperparameter Tuning**
* “Hyperparameter optimization in Python. Part 1: Scikit-Optimize.” by Jakub Czakon https://link.medium.com/Jhe60C6ccZ
* Bayesian Optimization (in detail): https://www.youtube.com/watch?v=vz3D36VXefI
*Bayesian Optimization:  https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0  
* Bayesian Optimization: https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a  
* 3D Visualization of Hyperparameters: https://towardsdatascience.com/using-3d-visualizations-to-tune-hyperparameters-of-ml-models-with-python-ba2885eab2e9  
* Converting HyperOpt to a class:  https://github.com/catboost/tutorials/blob/master/classification/classification_with_parameter_tuning_tutorial.ipynb
* Overriding functions at the instance level: https://stackoverflow.com/questions/394770/override-a-method-at-instance-level
* Deep Learning Hyperparametyer optimization: 
    - https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d
    = https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
* Parallelizing Hyperparameter optimization using PySpark clusters: https://towardsdatascience.com/100x-faster-randomized-hyperparameter-searching-framework-with-pyspark-4de19e44f5e6

**Genetic Algorithms**
* Introduction: https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad/ 
* Code Example: https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6  
* Full code: https://github.com/ahmedfgad/GeneticAlgorithmPython 
* Genetic Algorithm using tpot library: https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d

**Support Vector Machines (SVM)**
* Kernel Trick Visualization: https://www.youtube.com/watch?v=-Z4aojJ-pdg

**Bagging, Boosting, Stacking, Ensembles**
* Basic CART Algorihm: 
    - Explanation with Code: https://www.youtube.com/watch?v=LDRbO9a6XPU&feature=em-subs_digest
* Stacking
    - Library in Python to do Stacking: https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e
* XGBoost
    - TBD
* CatBoost
    - https://medium.com/@hanishsidhu/whats-so-special-about-catboost-335d64d754ae  

**Clustering**
* Various methods and Cluster Validation:  https://towardsdatascience.com/unsupervised-machine-learning-clustering-analysis-d40f2b34ae7e
* Spectral Clustering: https://towardsdatascience.com/spectral-clustering-82d3cff3d3b7 
* DBSCAN:  https://towardsdatascience.com/dbscan-algorithm-complete-guide-and-application-with-python-scikit-learn-d690cbae4c5d   

**Artificial Neural Networks**
* Optimization Algorithms: 
    - https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f
    - https://medium.com/analytics-vidhya/optimization-problem-in-deep-neural-networks-400f853af406

**Deep Learning**
* deeplearning.ai Notebooks 
    - (This is better): https://github.com/sanzgiri/deeplearning.ai
    - https://github.com/tamirlan1/Deeplearning.ai
* Stanford CS231
    - http://cs231n.github.io/
    - https://github.com/cs231n/cs231n.github.io

**Convoluted Neural Networks**


**Recurrent Neural Networks**
* https://towardsdatascience.com/an-introduction-to-recurrent-neural-networks-for-beginners-664d717adbd  

**Transfer Learning**
* https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

**Neural Network Architecture Search**
* https://papers.nips.cc/paper/7472-neural-architecture-search-with-bayesian-optimisation-and-optimal-transport.pdf 
* Code for above paper:  https://github.com/kirthevasank/nasbot   
* Google: Network Architecture Search using Genetic Algorithms:  https://ai.googleblog.com/2018/03/using-evolutionary-automl-to-discover.html

**Time Series**
* Why LSTM will not work for a random walk (stock) prediction problem: https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424
    - Very well presented article. Talks about the ARIMA theory and that the dataset after differencing is just white noise which by definition can not be predicted.
    - LSTM in this case becomes just a persistance model (prediction = last value) which is what ARIMA would also do.
* Why LSTM may not work for AR type of problems: https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/
* Using CNN type architecture by converting time series data to image: https://towardsdatascience.com/stock-market-action-prediction-with-convnet-8689238feae3
* Stock Market Prediction:
    - Using LSTM (1 layer) + 2 Dense layers: https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944
        * Uses stateful = True and works well over a longish run
        * Also uses a lot of data for training (> 13,000 sets of observations)
    - Using LSTM (4 layers) + 1 Dense layer: https://blog.usejournal.com/stock-market-prediction-by-recurrent-neural-network-on-lstm-model-56de700bff68
        * Uses stateful = False and not sure how well it predicts over the long run (test results were only shown for 20 days). 
        * Training data size ~ 5 years of daily returns ~ 1250 observations.
* InceptionTime for Time Series Classification: https://towardsdatascience.com/deep-learning-for-time-series-classification-inceptiontime-245703f422db


**Huber Loss Function**
* https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3

**Deploying Models**
* https://www.linkedin.com/posts/activity-6575387094733123584-cpHY/
    - Deploy an ML model ( #python scikit-learn ) as a REST API -> https://lnkd.in/esyH863
    - Deploy a Keras model ( #python keras ) -> https://lnkd.in/eSyvcg6
    - Deploy a Keras model repo -> https://lnkd.in/ev9iAzP
    - Deploy an R model using Docker -> https://lnkd.in/eSEAwYk
    - Deploy a tensorflow model on Android -> https://lnkd.in/eDG8ejM

## Python

**Nice Tutorials**
* large Collection of notebooks (inc. **Deep Learning**): https://github.com/donnemartin/data-science-ipython-notebooks/tree/master/deep-learning

**Writing Python Package**
* Adding Dependencies (example): https://github.com/eriklindernoren/Keras-GAN

**Auto ML**
* https://medium.com/thinkgradient/automated-machine-learning-an-overview-5a3595d5c4b5  
* autosklearn:  https://automl.github.io/auto-sklearn/master/#  
* MLBox: 
* TPOT:
* Auto-Keras: 
* HungaBunga (NEW, has issues - see eval) https://github.com/ypeleg/HungaBunga/blob/master/Readme.md

**Anomaly Detection**
* With LSTM in Keras: https://towardsdatascience.com/anomaly-detection-with-lstm-in-keras-8d8d7e50ab1b
* In Time Series Data:  https://towardsdatascience.com/anomaly-detection-time-series-4c661f6f165f

**Running from Command Line**
* Command Line Arguments: https://levelup.gitconnected.com/the-easy-guide-to-python-command-line-arguments-96b4607baea1

**Graph Algoriths**
* https://www.kdnuggets.com/2019/09/5-graph-algorithms-data-scientists-know.html

**Interactivity**
* Interactive Widgets: https://ipywidgets.readthedocs.io/en/latest/

**Dashboards and Apps**
* Dash App Development (from plotly)
    * https://dash.plot.ly/: Basics are covered in tutorials 1-3
    * Sample Apps: https://github.com/plotly/dash-sample-apps/tree/master/app
    * Detailed Example: https://towardsdatascience.com/how-to-build-a-complex-reporting-dashboard-using-dash-and-plotl-4f4257c18a7f
* Starlette
* Streamlit
* Flask


### Neural Networks and Deep Learning 

**Fast AI**
* Local Installation
    - v1: https://medium.com/@pierre_guillou/how-to-install-fastai-v1-on-windows-10-ca1bc370dce4
    - Older version (0.7): https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652
* Cyclical Learning Rate (CLR): 
    - https://iconof.com/1cycle-learning-rate-policy/
    - CLR Paper: https://arxiv.org/pdf/1506.01186.pdf
    - Implementing CLR in Keras: https://github.com/bckenstler/CLR
* Superconvergence using CLR: https://arxiv.org/pdf/1708.07120.pdf
* APIs:
    - DataBlock API: https://docs.fast.ai/data_block.html
    - Vision API: https://docs.fast.ai/vision.image.html
    - Datasets: https://course.fast.ai/datasets

**Convolutional Neural Networks**
* Image Processing Pipeline: https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Keras_flow_from_directory.ipynb
* Image Processing Comprehensive Guide: https://www.pyimagesearch.com/start-here/

**Keras vs. PyTorch**
* Comparison: 
    - Overview: https://deepsense.ai/keras-or-pytorch/
    - Model definition options and fit comparison: https://towardsdatascience.com/pytorch-vs-tensorflow-in-code-ada936fd5406
    - Code comparison: https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983

**Keras Related Topics**
* Early Stopping: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
* Grid Search for Time Series RNN Architecture Search: https://machinelearningmastery.com/how-to-grid-search-deep-learning-models-for-time-series-forecasting/


**Deep Learning Examples**
* PyTorch
    - Basic: https://towardsdatascience.com/training-a-neural-network-using-pytorch-72ab708da210
    - YOLO v3 in PyTorch: https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
    - CIFAR Image Classification in PyTorch: https://deepsense.ai/deep-learning-hands-on-image-classification/
    - Visualization CNN: https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
        * Notebook available in my Colab: https://colab.research.google.com/drive/1WSCuLiFSpUBRNu4WkN77cuBWKieTk-VT)
* Keras
    - Face Recognition: https://chatbotslife.com/how-to-build-a-face-recognition-system-17f8425da5e3
    - Time Series
        * Basic: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
        * Multivariate: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        * Formatting time series data in the right format for Keras: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
        
* Misc
    - https://d2l.ai/chapter_computer-vision/index.html (An interactive deep learning book with code, math, and discussions. Used at Berkeley, University of Washington and more) Link from @David Stroud

**Deep Learning Datasets**
* MNIST: https://keras.io/datasets/
* notMNIST: https://github.com/stared/keras-mini-examples/blob/master/notMNIST_starter.ipynb
* CIFAR: https://keras.io/datasets/
* Creating your own Image Datasets
    - (**Preferred**) Using fast.ai APIs https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb (works in Chrome, may not work in Firefox)
    - Using Google Chrome Extension: https://towardsdatascience.com/https-medium-com-drchemlal-deep-learning-tutorial-1-f94156d79802
    - https://debuggercafe.com/create-your-own-deep-learning-image-dataset/

**RNN**
* Text Generation using RNNs
    - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    - Character level RNN model: https://gist.github.com/karpathy/d4dee566867f8291f086
    - Text Generation (like Shakespeare): https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

**Deep Learning with GPUs**
* Keras on Amazon EC2: https://medium.com/hackernoon/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac



## R 

**Comaprison to Python**
* Code comparison Cheatsheet: https://towardsdatascience.com/cheat-sheet-for-python-dataframe-r-dataframe-syntax-conversions-450f656b44ca

**R Utility Packages**
* dtplyr: https://www.business-science.io/code-tools/2019/08/15/big-data-dtplyr.html  

**Tidyverse**
* purrr: https://purrr.tidyverse.org/  
* dplyr for Python: https://github.com/allenakinkunle/dplyr-style-data-manipulation-in-python  

**Visualizations**
* https://bbc.github.io/rcookbook/#how_to_create_bbc_style_graphics
* Organizing ggplots using Patchwork package (very good): https://github.com/thomasp85/patchwork

**Tidyquant (importing historic stock data)**
* https://github.com/business-science/presentations/blob/master/2019_05_17_RFinance_Tidyquant_Portfolio_Optimization/R_Finance_tidyquant_matt_dancho.pdf
* https://www.youtube.com/watch?v=OjIZIHPwvKs&feature=youtu.be 
* Core Functions: https://cran.r-project.org/web/packages/tidyquant/vignettes/TQ01-core-functions-in-tidyquant.html
* Portfolio Performance Analysis: https://cran.r-project.org/web/packages/tidyquant/vignettes/TQ05-performance-analysis-with-tidyquant.html

**Plotting (ggplot)**
* 3D plots with ggplot2: https://www.rayshader.com/reference/plot_gg.html 

**Time Series**
* Forecasting wth RNN: https://blogs.rstudio.com/tensorflow/posts/2017-12-20-time-series-forecasting-with-recurrent-neural-networks/

**Caret**
* Overwriting caret's random search: https://stackoverflow.com/questions/53716810/how-to-random-search-in-a-specified-grid-in-caret-package

**MLR**  Needs cleanup
* General sampling stuff: https://mlr.mlr-org.com/articles/tutorial/resample.html
* Basic tuning: https://mlr.mlr-org.com/articles/tutorial/tune.html
* Advanced tuning: https://mlr.mlr-org.com/articles/tutorial/advanced_tune.html
* So for example custom models https://mlr.mlr-org.com/articles/tutorial/create_learner.html
* And a bunch of examples of wrapping and extending here https://mlr.mlr-org.com/articles/tutorial/wrapper.html (the bulleted links), basically you can use the wrapper class to really extend the functionality

**Graphs in R**
* DiagrameR package: http://visualizers.co/diagrammer/

**Creating a R Package**
* https://evamaerey.github.io/package_in_20_minutes/package_in_20_minutes
* How to: https://hilaryparker.com/2014/04/29/writing-an-r-package-from-scratch/
* Example from David Josephs: https://github.com/josephsdavid/tswgewrapped/blob/master/R/aicbic.R 
* Specific Commands
    - usethis::use_test() # To use test_that with your package
    - devtools::test()    # to run your unit tests
    - covr::report()      # creating a unit test coverage report
    - usethis::use_vignette("vignette_name")  # create a vignette to show how the object flow works
    - usethis::use_pipe() # To remove warnings due to the pipe operation from magrittr

**Finance**
* Discounted Cash Flow Analysis
    - Matt Dancho: https://www.business-science.io/finance/2020/02/21/tidy-discounted-cash-flow.html 

**Publishing**
* Books and Articles: 
    - https://bookdown.org/yihui/bookdown/
    - **Recommended Readings for manual: https://bookdown.org/yihui/bookdown/structure-of-the-book.html**
* Blogs and Websites: 
    - https://bookdown.org/yihui/blogdown/
    - **Recommended Workflow: https://bookdown.org/yihui/blogdown/workflow.html**
  
**Snippets or Macros**
* Overview: https://support.rstudio.com/hc/en-us/articles/204463668-Code-Snippets
* Useful Snippets: https://raw.githubusercontent.com/gadenbuie/snippets/master/r.snippets

## Others

**Interview Guide**
* https://medium.com/better-programming/the-data-science-interview-study-guide-c3824cb76c2e  

**Resume Examples**
- https://www.slideshare.net/slideshow/embed_code/key/kuZh6Pj9fhETX4


