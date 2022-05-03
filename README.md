<h1 align="center">
Unsupervised Computational Framework for Drug Repurposing
</h1>

<!--
This is the official repository for the paper [L. Menestrina, M. Recanatini, Unsupervised Computational Framework for Drug Repurposing: Application to two Neurological Disorders](url)  
[doi](doi)  
It contains the Python 3 implementation of the pipeline, as well as the results of its application on Huntington's disease and multiple sclerosis    
-->
&nbsp;

<p align="center">
  <img src="https://github.com/LucaMenestrina/UnsupervisedComputationalFrameworkForDrugRepurposing/master/graphical_abstract.svg" alt="Graphical Abstract - Pipeline" width="90%"/>
</p>


### Abstract

Drug repurposing consists in identifying additional uses for known drugs and, since these new findings are built on previous knowledge, it reduces both the length and the costs of the drug development. In this work, we assembled an automated computational pipeline for drug repurposing, integrating also a network-based analysis for screening the possible drug combinations. The selection of drugs relies both on their proximity to the disease on the protein-protein interactome and on their influence on the expression of disease-related genes. Combined therapies are then prioritized on the basis of the drugs’ separation on the human interactome and the known drug-drug interactions. We eventually collected a number of molecules, and their plausible combinations, that could be proposed for the treatment of Huntington’s disease and multiple sclerosis. Furthermore, this pipeline could potentially provide new suggestions also for other complex disorders.  

**Keywords:** Computational drug discovery, Drug repurposing, Network pharmacology, Network analysis, Huntington’s disease, Multiple sclerosis  

<!--
**Bibtex**  

Lorem Ipsum  
-->

### Usage

**Download** from GitHub:  
```
wget https://github.com/LucaMenestrina/UnsupervisedComputationalFrameworkForDrugRepurposing/archive/refs/heads/master.zip
```

**Install** required packages:  
```
pip install -r requirements.txt
```

**Perform Analysis**

```
python3 run.py [-h] [-d DISEASE] [-g GENES [GENES ...]] [-gf GENES_FILE]

optional arguments:
  -h,  --help                                       show this help message and exit
  -d,  --disease DISEASE                            Disease Name
  -g,  --genes GENES [GENES ...]                    Disease-Related Genes (separated by a space)
  -gf, --genes_file GENES_FILE                      File Containing Disease-Related Genes (one per line)
  -cl, --cell_lines CELL_LINES [CELL_LINES ...]     Base Disease-Related Cell Lines in LINCS Database (separated by a space)
```
Specify only one of 'genes' and 'genes_file'
If no Disease Name is provided the analysis will be performed for Huntington's Disease and Multiple Sclerosis as in the paper

**TO NOTICE:**  
Some Sources (DrugBank, DisGeNET, OMIM) _require identification_ for downloading their files.  
Set the environmental variables: ```DRUGBANK_EMAIL```, ```DRUGBANK_PASSWORD```, ```DISGENET_EMAIL```, ```DISGENET_PASSWORD```, and ```OMIM_APIKEY``` or save them in a hidden file ```.env``` in the main directory of your project.

### Sources

- [APID](http://cicblade.dep.usal.es:8080/APID/init.action)
- [BioGRID](https://thebiogrid.org/)
- [DisGeNET](https://www.disgenet.org/)
- [GO](http://geneontology.org/)
- [HGNC](https://www.genenames.org/)
- [HPO](https://hpo.jax.org/)
- [HuRI](http://www.interactome-atlas.org/)
- [InnateDB](https://www.innatedb.com/)
- [INstruct](http://instruct.yulab.org/)
- [IntAct](https://www.ebi.ac.uk/intact/home)
- [LINCS](https://lincsproject.org/)
- [NCBI](https://www.ncbi.nlm.nih.gov)
- [OMIM](https://www.omim.org/)
- [SignaLink](http://signalink.org/)
- [STRING](https://string-db.org/)

Additional info reported in [sources.json](https://github.com/LucaMenestrina/UnsupervisedComputationalFrameworkForDrugRepurposing/master/data/sources/sources.json)
