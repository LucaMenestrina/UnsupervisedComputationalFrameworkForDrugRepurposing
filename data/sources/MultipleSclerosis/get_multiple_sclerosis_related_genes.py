import pandas as pd
import re
import requests

import databases as dbs

ncbi = dbs.NCBI()

__all__ = ["multiple_sclerosis_genes"]

### MULTIPLE SCLEROSIS

# Kegg doesn't have associated pathways
kegg_genes = []  # No associated genes

omim = dbs.OMIM(only_query=True)

omim_genes = omim.get_disease_genes("Multiple Sclerosis")

# PheGenI (https://www.ncbi.nlm.nih.gov/gap/phegeni)  P-value <1x10^-8
phegeni_associations = pd.read_csv(
    "data/sources/MultipleSclerosis/PheGenI_Association.tab",
    sep="\t",
    usecols=["Gene", "Gene 2"],
)

phegeni_genes = (
    phegeni_associations["Gene"].tolist() + phegeni_associations["Gene 2"].tolist()
)

# DISEASES (https://diseases.jensenlab.org/Entity?order=textmining,knowledge,experiments&textmining=10&knowledge=10&experiments=10&type1=-26&type2=9606&id1=DOID:2377)  Only manually curated
diseases_genes = ["TNFRSF1A", "IL2RA", "HLA-DRB1", "IL7R", "CYP27B1"]

disgenet = dbs.DisGeNET()
disgenet_genes = disgenet.database.query(
    "EI >= 0.95 & diseaseName == 'Multiple Sclerosis'"
)["geneSymbol"].tolist()

multiple_sclerosis_genes = set(
    kegg_genes + omim_genes + phegeni_genes + diseases_genes + disgenet_genes
)

multiple_sclerosis_genes = {
    ncbi.check_symbol(gene): ncbi.get_id_by_symbol(gene)
    for gene in multiple_sclerosis_genes
    if ncbi.get_id_by_symbol(gene) and ncbi.check_symbol(gene)
}

pd.DataFrame(
    ((symbol, id) for symbol, id in multiple_sclerosis_genes.items()),
    columns=["geneSymbol", "geneId"],
).to_csv("data/sources/MultipleSclerosis/multiple_sclerosis_genes.tsv", sep="\t")
