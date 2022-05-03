import pandas as pd
import re
import requests
import databases as dbs

ncbi = dbs.NCBI()

__all__ = ["huntington_genes"]

### HUNTINGTON DISEASE

# Kegg (https://www.genome.jp/dbget-bin/www_bget?pathway:hsa05016)
dfs = pd.read_html(
    "https://www.genome.jp/dbget-bin/www_bget?pathway:hsa05016", match="hsa05016"
)
kegg_genes = [
    ncbi.get_symbol_by_id(id)
    for id in re.findall(
        "([0-9]+)\ [A-Z 0-9 \-]+;", dfs[3][dfs[3][0] == "Gene"][1].values[0]
    )
]

omim = dbs.OMIM(only_query=True)

omim_genes = omim.get_disease_genes("Huntington Disease")


# PheGenI (https://www.ncbi.nlm.nih.gov/gap/phegeni)
phegeni_genes = []  # No associated genes

# DISEASES (https://diseases.jensenlab.org/Entity?order=textmining,knowledge,experiments&textmining=10&knowledge=10&experiments=10&type1=-26&type2=9606&id1=DOID:12858)   Only manually curated
diseases_genes = ["HTT"]

disgenet = dbs.DisGeNET()
disgenet_genes = disgenet.database.query(
    "EI >= 0.95 & diseaseName == 'Huntington Disease'"
)["geneSymbol"].tolist()

huntington_genes = set(
    kegg_genes + omim_genes + phegeni_genes + diseases_genes + disgenet_genes
)

huntington_genes = {
    ncbi.check_symbol(gene): ncbi.get_id_by_symbol(gene)
    for gene in huntington_genes
    if ncbi.get_id_by_symbol(gene) and ncbi.check_symbol(gene)
}

pd.DataFrame(
    ((symbol, id) for symbol, id in huntington_genes.items()),
    columns=["geneSymbol", "geneId"],
).to_csv("data/sources/Huntington/huntington_genes.tsv", sep="\t")
