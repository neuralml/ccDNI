# Cerebro-cerebellar RNN

Code used for paper [Cerebro-cerebellar networks facilitate learning through feedback decoupling](https://www.nature.com/articles/s41467-022-35658-8) (see also [Cortico-cerebellar networks as decoupling neural interfaces](https://proceedings.neurips.cc/paper/2021/hash/3ffebb08d23c609875d7177ee769a3e9-Abstract.html)).

## Dependencies
Beyond the standard python libraries, you will need [Pytorch](https://pytorch.org/) (we use version 1.6.0, but later should work) to define the neural network models and as well as [ignite](https://github.com/pytorch/ignite) (we use version 0.2.1) which wraps the training regime. 

## Steps to run 
For the **linedraw** and **seqmnist**-based tasks:
1. Go into the /scripts folder and choose the corresponding folder according to the experiment you want to run. <br>
Open the *train-model.py* file. 
2. In the file, manually define the path to the src folder (e.g. Documents/ccDNI/src) and where to save results (e.g. Documents/ccDNI/results) for your system. These are defined in the *src_path* and *root* variables respectively. 
3. *To run within IDE*:
   * Select the hyperparameters of the experiment (see main ones below). E.g. to run the ccRNN model for 5 epochs set args.model = 'DNI_LSTM' and args.epochs = 5
   * Run the file (press play button).
   *To run from terminal*
   * Comment out suggested hyperparameter values (underneath where the args variable is defined)
   * Set the *train-model.py* file location as the current working directory
   * Run the python command on the file with desired arguments - e.g. *python train-model.py -model=DNI_LSTM -epochs=5*
4. Check the results are saved in the folder defined in step 2. The resulting *numpy* file should have shape *(a, b, c, d)*, where *a* is the number of number of seeds, *b* is the number of models (usually just one), *c* is the number of different metrics (e.g. train and validation MSE), and *d* is the number of epochs.

For the **image captioning** task: 
1. Go into the /other/image-captioning folder. 
2. Run the bash file *download.sh* to download the dataset with the command *./download.sh*. 
3. Preprocess the data by running the *build_vocab.py* and *resize.py* scripts.
4. Go into the *train.py* file. Configure data paths (i.e. *vocab_path*, *image_dir*, *caption_path*) based on destinations of steps 2, 3 above. Configure path to save model (*model_path*). Replace also the *save_path* variable defined in the *get_fn()* method to where you wish the model losses to be saved.
5. Run the *train.py* file with the desired experiment hyperparameters (either directly in IDE or via terminal, see above).
6. Check/plot model losses are saved in the *save_path* defined in 4. 
7. To sample a caption for an example image post training, run the *sample.py* file with the *decoder_path* set as the filepath of the trained model.

## Plot results
Example plotting code can be found in the /plotting directory. 
To the learning curves and trained model output for the simple line drawing task as shown in Figure 2 in the [Neurips](https://proceedings.neurips.cc/paper/2021/hash/3ffebb08d23c609875d7177ee769a3e9-Abstract.html) paper, run the *linedraw_plotting.py* file (to be added soon). 
To plot the learning curves under different levels of cerberal feedback (i.e. backpropagation truncation sizes) for the seqmnist line-drawing and digit-drawing tasks as in Figure S3 in Neurips, run the *seqmnist_curves_all.py* file. 

## Main experiment hyperparameters 
The primary hyperparameters with which we modify our experimental setup can be listed as 
* *model* - type of model used. Can be standard 'LSTM' (cRNN) or DNI enabled 'DNI_LSTM' (ccRNN)
* *bptt* - cerebral temporal feedback (when normalised by sequence length of task)/truncation size. 
* *spars-int* - degree of sparseness in teacher feedback for task. E.g. for Fig 2 in Neurips *spars-int=2* so that teacher feedback is only available every other timestep.

For the seqmnist-based drawing tasks:
* *fixed-mag* - Fixed magnitute for each line. Set as true to do line-drawing task.
* *digit-draw* - set as true to do digit-drawing task




Pytorch experiment structured according to https://github.com/miltonllera/pytorch-project-template <br>
The DNI implementation is based on https://github.com/koz4k/dni-pytorch 


<b>References:</b><br>
Boven, Pemberton et al. bioRxiv, https://www.biorxiv.org/content/10.1101/2022.01.28.477827v1 <br>
Pemberton, Boven et al NeurIPS 2021, https://proceedings.neurips.cc/paper/2021/hash/3ffebb08d23c609875d7177ee769a3e9-Abstract.html
