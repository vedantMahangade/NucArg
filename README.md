# NucArg: Antibiotic Resistance Detection
A transformer based model to detect antibiotic resistant genes

This repository hosts scripts for training two nucleotide models to detect antibiotic resistant genes (ARGs). The models are trained on long read (>100 bp) and short read DNA sequences (100 bp) to predict antibiotic resistance classes. The dataset is based on a research paper:
[DeepARG: a deep learning approach for predicting antibiotic resistance genes from metagenomic data](https://pmc.ncbi.nlm.nih.gov/articles/PMC5796597/)
This paper uses a deep learning approach on amino acid data and implements a concept of dissimilarity based classification.

NucArg is based on Nucleotide (DNA) sequences therefore the original Amino Acid dataset from the referenced paper was repurposed to create a Nucleotide dataset. NucArg supports a total of 26 Antibiotic classes, including non-resistant (negative) class which was not in the referenced paper.

**The classes are**:
beta_lactam, bacitracin, chloramphenicol, aminoglycoside, macrolide-lincosamide-streptogramin, multidrug, polymyxin, fosfomycin, sulfonamide, glycopeptide, mupirocin, trimethoprim, quinolone, fosmidomycin, tetracycline, rifampin, peptide, kasugamycin, qa_compound, fusidic_acid, tunicamycin, tetracenomycin, streptothricin, thiostrepton, aminocoumarin, non_resistant

The short read dataset is created by spliting the long reads in to chunks of 100. (*Note: There might be some classes missing in the short read dataset due to issues in processing*)

## Supervised Fine-Tuning
The pretrained model used here is [InstaDeepAI/nucleotide-transformer-v2-100m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species). 
This model uses ESM architecture and is pretrained on collection of 850 genomes, representing a total of 174B nucleotides making it ideal for tasks like SequenceClassification.

The limitation of this model is the max_lenght of sequence allowed, which is 2048, while some sequences in the long read dataset exceeded this limit. To make training more efficient and make sure model learns efficiently three steps where taken:

- Mixed precision training - fp16
- Effective Batch Size: 64
    - Using batch size of 8
    - Gradient accumulation steps of 8 for each iteration
- Sequence splitting
    - max_length of 1024 with
    - overlap of 512

The fine-tuned models are pushed to huging face:

[NucArg Long Reads Model](https://huggingface.co/vedantM/NucArg_LongRead)

[NucArg Short Reads Model](https://huggingface.co/vedantM/NucArg_ShortRead)



## Other items to consider
The models have huge potential for improvement and optimization. 
Ablation tests in *[Ablation testing - Long Read - Logistic Regression.ipynb]()* and *[Ablation testing - Short Read - Logistic Regression.ipynb]()* prove that the fine-tuning was successfull. Additional data could definetly help improve model's performance and help it generalize more on unseen classes as well.

The script *[Ablation testing - Long Read - Neuronal.ipynb]()* performs neuronal ablation and the results indicate much of the model's layer are redundunt and the model has high potential for optimization through pruning.

## Deployment
Last but not least, a streamlit app that inferences both short read and long read model has been deployed to hugging face spaces:

[NucArg Streamlit App](https://huggingface.co/spaces/vedantM/NucArg)

Feel free to play arround with the model and the app. Do reachout for any suggestions and improvements.
