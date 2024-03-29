# Drug_Discovery-COVID-19
**Small Molecule Development using AI**

So...I approached this problem in a rather unorthodox way. Normally, you would decide to use a Autoencoder or an LSTM Network but I treated this problem as a Text-generation problem because the first step is to generate SMILE-Strings. After, that they were visualized in [RDKit](https://www.rdkit.org/docs/) and then tested in [PyRx](https://pyrx.sourceforge.io/). I used [OpenAI's GPT2](https://openai.com/blog/better-language-models/) to speed up the whole process. 

**NOTE:** The code was made and run in **Google Colab**. Click [here](https://www.youtube.com/watch?v=inN8seMm7UI) to learn more.

# About the files
* 6LU7.pdb - the protease of the novel corona virus (macromolecule).
* COVID-19_Old.py - This is was my first approach, where everything was done from scratch. You can have a look at it, but I     don't recommend you to run it. 
* COVID_19.ipynb - Compounds are generated here.
* Smiles_Testing.ipynb - Compounds are visualized here.
* Final_Results.ipynb - the top 3 drugs are visualized here. 
* dataset_for_training.csv - the dataset used for training.
* compounds_generated.csv - the final 10 compounds are stored here.
* compounds_generated.sdf - converted the compounds_generated.csv to an .sdf file so that it can be processed by PyRx.
* Final_Results.csv - here you will find the final results i.e. the binding affinity of all the drugs. 

# Prerequisites
In my opinion, AI's biggest application is Healthcare. Computaional Drug Design is one area in Healthcare where AI can have one of its biggest impacts. 

All drugs can be represented in the form of SMILES. SMILES in Molecular Biology stands for **S**implified **M**olecular-**I**nput **L**ine-**E**ntry **S**ystem. It is just a way to represent compounds in text format. So, our goal for the project should be to find the best drug that can work on the Novel Corona Virus [Protease](https://en.wikipedia.org/wiki/Protease). 

Recently, the [Protein Data Bank (PDB)](https://www.rcsb.org/), released the protease of the Novel Corona Virus - [6LU7](https://www.rcsb.org/structure/6LU7). 

When you're is testing how well a drug can act on a virus, one thing you want to look for is the [Binding Affinity](https://www.google.com/search?rlz=1C5CHFA_enIN848IN849&sxsrf=ALeKk01_96Hq1mXH46Ub6pV8xCBx7V0O5Q%3A1587805320880&ei=iPyjXsKoNaTDpgeM9YqICg&q=binding+affinity+meaning&oq=binding+affinity+meaning&gs_lcp=CgZwc3ktYWIQAzICCAAyBggAEBYQHjoECAAQRzoECAAQQzoFCCEQoAFQvBFY5Dlg2jtoB3ACeACAAbQBiAGZC5IBAzAuOZgBAKABAaoBB2d3cy13aXo&sclient=psy-ab&ved=0ahUKEwjCjP7gm4PpAhWkoekKHYy6AqEQ4dUDCAw&uact=5) of that drug.

A "Drug" in Biochemistry terms is called a [Ligand](https://en.wikipedia.org/wiki/Ligand_(biochemistry)) and a the virus can be called a [Macromolecule](https://en.wikipedia.org/wiki/Macromolecule). So, in simple terms, Binding Affinity can be described as to how well a ligand can bind onto the protease of a macromolecule and act on it. This action of a ligand binding onto a macromolecule is called [Docking](https://en.wikipedia.org/wiki/Docking_(molecular)). 

![](https://github.com/JustHarsh/Drug_Discovery-COVID-19/blob/master/Docking_representation_2.png)
- *credits go to Wikipedia. Image can be found [here](https://en.wikipedia.org/wiki/Docking_(molecular))*

# Requirements 
* Tensorflow 1.x
* GPT-2
* PyRx
* DataWarrior

# Data 
I tried using the [ChemBl Dataset](https://www.ebi.ac.uk/chembl/) for my project. However, I was short on compute power and time and therefore ended up using the results from another extraordinary repository that worked on the same project as mine. You can check it out [here](https://github.com/mattroconnor/deep_learning_coronavirus_cure). Now why was this a good idea? Well, Matt (person who owns the repo) generated those drugs from the ChemBl and the Moses dataset. So essentially, I did the work of improving on his results (if you think critically). My honest recommendation would be to use the ChemBl dataset if you want to work it out from scratch; else you might as well use my or Matt's results to get your project working. 

So officially, I give Matt R O'Connor **ALL the credits** for the training data used in the project. 

# Approach 
Since this project was run in Google Colab, you don't need to install anything locally. All you need is a computer, an active internet connection and lots patience, dedication and perseverance. Just open a Colab Notebook, and follow whatever's written in the **COVID-19.ipynb**. You should start by uninstalling Tensorflow 2.x and install Tensorflow 1.x since GPT2 works very smoothly with TF 1.x. 

Copy the generated SMILES to a file in your local system. Later, go to the **Smiles_Testing.ipynb** notebook and visualize the generated SMILES. Honestly speaking, more than half the SMILES generated will have errors in them. That is why I suggest you to first copy them to a .txt file and then copy the final SMILES to a .csv file. I used [DataWarrior](http://www.openmolecules.org/datawarrior/download.html) mainly to convert a .csv file to a .sdf file to test in PyRx. 

After that, I used the .sdf file for docking with 6LU7. After achieving the results, They were converted to a .csv file. You can find the results in **Final_Results.csv**. 

To find how to perform docking in PyRx watch this [tutorial](https://www.youtube.com/watch?v=2t12UlI6vuw). 

# What's next?
These drugs should be verified by an expert in the field of research to verify the synthetic feasibilty of the drugs. They should then be pushed for clinical trials and then be used for treating patients. 

# Challenges
I failed about 6 times before I finally solved this problem. My failures were due to a lack of clarity about where to get a good dataset from, what technique I should use to discover new drugs that could potentially cure something like COVID-19 and how to comprehend the results I got. After multiple unsuccessful attempts of solving this problem from scratch on my local computer, I realized that I can use a [pre-trained model](https://www.youtube.com/watch?v=Ui1KbmutX0k&t) and treat this problem as a text-generation problem. The results I obtained were fairly conclusive, and the best for the training time and computational power I had available running on Google Colab.
