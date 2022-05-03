import pandas as pd
import numpy as np
import re
from collections import namedtuple
from tqdm import tqdm
from utils import profile
from statsmodels.stats.multitest import multipletests

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "IGSEA"
log = logging.getLogger(logging_name)


################################################################################
#
#                   Inverted Gene Set Enrichment Analysis
#
################################################################################


from cmapPy.pandasGEXpress.parse import parse as parse_gctx
import gseapy
from databases import LINCS

prerank_gsea_results = namedtuple(
    typename="IGSEA_results", field_names=["ES", "NES", "pvalue"],
)


# @profile()
def prerank_gsea(
    expression_df, disease_name, gene_set, lincs, nperm=int(1e5), seed=12345
):
    """
        Gens Set Enrichment Analysis for a Preranked Single Signature
    """
    log = logging.getLogger("IGSEA:prerank_gsea")
    rank = expression_df.sort_values(ascending=False)
    gsea_results, _, _, _ = gseapy.algorithm.gsea_compute(
        data=rank,
        n=nperm,
        gmt={
            f"{lincs.sigid2DBid[rank.name]}_{disease_name.replace(' ', '')}": gene_set
        },
        weighted_score_type=1,
        permutation_type="gene_set",
        method=None,
        pheno_pos="Pos",
        pheno_neg="Neg",
        classes=None,
        ascending=False,
        processes=1,
        seed=seed,
    )

    ES, nES, pvalue, _ = tuple(*gsea_results)

    return prerank_gsea_results(ES, nES, pvalue)


# @profile()
def IGSEA(
    disease_name, gene_set, lincs, nperm=int(1e5), alpha=0.25, seed=12345,
):
    """
        Inverted Gene Set Enrichment Analysis
        As in the paper https://academic.oup.com/bioinformatics/article/36/17/4626/5855131
    """
    log = logging.getLogger("IGSEA:igsea")

    gene_set = tuple({str(id) for id in gene_set} & lincs.BING_genes)

    igsea_results = {
        col: prerank_gsea(
            database_batch.iloc[:, col_n],
            disease_name,
            gene_set,
            lincs,
            nperm=nperm,
            seed=seed,
        )
        for database_batch in lincs.database
        for col_n, col in tqdm(enumerate(database_batch.columns))
    }

    # Significance analysis
    pvalues = np.array([result.pvalue for result in igsea_results.values()])
    mask, pvalues_corrected, _, _ = multipletests(
        pvals=pvalues, alpha=alpha, method="fdr_bh"
    )

    results_df = pd.DataFrame(
        {
            "Signature": igsea_results.keys(),
            "DrugBank_ID": [lincs.sigid2DBid.get(id) for id in igsea_results.keys()],
            "DrugBank_Name": [
                lincs.sigid2DBname.get(id) for id in igsea_results.keys()
            ],
            "Enrichment_Score": [result.ES for result in igsea_results.values()],
            "Normalized_Enrichment_Score": [
                result.NES for result in igsea_results.values()
            ],
            "p-value": [pvalue for pvalue in pvalues],
            "FDR": [
                pvalue for pvalue in pvalues_corrected
            ],  # corrected p-value for multiple tests (False Discovery Rate)
            "Cell_Line": [lincs.sigid2cell.get(id) for id in igsea_results.keys()],
        }
    )

    results_df.sort_values(
        by=["p-value", "Normalized_Enrichment_Score"],
        ascending=[True, False],
        inplace=True,
        key=abs,
        ignore_index=True,
    )

    results_df["p-value"] = [
        pvalue if pvalue != 0 else f"<{1/nperm}" for pvalue in results_df["p-value"]
    ]
    results_df["FDR"] = [
        fdr if fdr != 0 else f"<{1/nperm}" for fdr in results_df["FDR"]
    ]

    results_df.to_csv(
        f"data/results/{disease_name.replace(' ', '')}/IGSEA_results.tsv",
        sep="\t",
        index=False,
    )

    return results_df


if __name__ == "__main__":
    # from data.sources.Alzheimer.get_alzheimer_related_genes import alzheimer_genes
    from data.sources.Huntington.get_huntington_related_genes import huntington_genes
    from data.sources.MultipleSclerosis.get_multiple_sclerosis_related_genes import (
        multiple_sclerosis_genes,
    )

    for disease_name, gene_set, base_cell_lines in (
        # ("Alzheimer", {str(id) for id in alzheimer_genes.values()}, ["NEU", "NPC"]),
        (
            "Huntington",
            {str(id) for id in huntington_genes.values()},
            ["NEU", "NPC", "SHSY5Y", "SKB", "SKL"],
        ),
        (
            "Multiple Sclerosis",
            {str(id) for id in multiple_sclerosis_genes.values()},
            [
                "NEU",
                "NPC",
                "SHSY5Y",
                "HL60",
                "JURKAT",
                "NOMO1",
                "PL21",
                "SKM1",
                "THP1",
                "U937",
                "WSUDLCL2",
            ],
        ),
    ):
        log.info(f"Performing Inverted Gene Set Enrichment Analysis for {disease_name}")
        lincs = LINCS(base_cell_lines=base_cell_lines, batch_size=2500)
        igsea_results = IGSEA(disease_name, gene_set, lincs)
