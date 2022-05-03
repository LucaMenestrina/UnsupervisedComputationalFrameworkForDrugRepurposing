import logging

logging.basicConfig(level=logging.INFO)
logging_name = "drug_target_gene"
log = logging.getLogger(logging_name)


################################################################################
#
#                           Drug-Target-Gene Network
#
################################################################################


def get_drugs(disease_name, significance_threshold=0.25):
    import pandas as pd
    import numpy as np
    import re

    log = logging.getLogger("drug_target_gene:get_drugs")
    log.info(
        f"Retrieving proximal and significantly enriched (FDR < {significance_threshold}) drugs"
    )

    network_proximity_results = pd.read_csv(
        f"data/results/{disease_name.replace(' ', '')}/network_proximities.tsv",
        sep="\t",
        index_col=0,
        skiprows=3,
    )
    igsea_results = pd.read_csv(
        f"data/results/{disease_name.replace(' ', '')}/IGSEA_results.tsv",
        sep="\t",
        index_col=0,
    )
    with open(
        f"data/results/{disease_name.replace(' ', '')}/network_proximities.tsv"
    ) as infile:
        infile.readline()
        distance_threshold = float(
            re.findall("\# Threshold: ([0-9]\.[0-9]{1,2})", infile.readline())[0]
        )
    proximal_igsea_results = igsea_results[
        igsea_results["DrugBank_ID"].isin(
            set(
                network_proximity_results[
                    network_proximity_results["Distance"] < distance_threshold
                ].index
            )
        )
    ].copy()

    significant_proximal_igsea_results = proximal_igsea_results[
        proximal_igsea_results["FDR"].apply(
            lambda fdr: True
            if isinstance(fdr, str) and "<" in fdr
            else float(fdr) < 0.25
        )
    ]

    return significant_proximal_igsea_results.merge(
        network_proximity_results.reset_index()[
            ["DrugBank_ID", "Distance", "Proximity"]
        ],
        how="left",
        on="DrugBank_ID",
    ).copy()


def draw_drug_target_gene_network(disease_name, disease_genes):
    import pandas as pd
    import networkx as nx
    import seaborn as sns
    from pyvis.network import Network
    import matplotlib as mpl
    from databases import DrugBank

    log = logging.getLogger("drug_target_gene:draw_drug_target_gene_network")

    drugbank = DrugBank()

    drugs = get_drugs(disease_name)

    drugbank_drug_targets = {
        k: list(ts)
        for k, ts in {
            d.id: {
                t.symbol
                for t in d.targets
                if t.type == "protein" and t.organism == "Humans" and t.symbol != None
            }
            for d in drugbank.database
        }.items()
        if ts
    }

    ppi = pd.read_csv(
        "data/sources/interactome_slim.tsv.gz", sep="\t", compression="gzip"
    )
    interactome = nx.from_pandas_edgelist(ppi, source="source", target="target")
    interactome = interactome.to_undirected()

    drugs = set(drugs.drop_duplicates(subset=["DrugBank_ID"])["DrugBank_ID"].values)

    drug2target = {
        (drug, target) for drug in drugs for target in drugbank_drug_targets[drug]
    }
    targets = {edge[1] for edge in drug2target}
    related_genes = {gene for gene in disease_genes}
    proteins = related_genes.union(targets)
    protein2protein = {
        (source, target)
        for source in proteins
        if source in interactome.nodes
        for target in interactome.neighbors(source)
        if target in proteins and source != target
    }

    G = nx.Graph()
    G.add_nodes_from(proteins.union(drugs))
    G.add_edges_from(drug2target.union(protein2protein))

    nx.write_adjlist(
        G,
        f"data/results/{disease_name.replace(' ', '')}/drug_target_gene_network.adjlist",
    )

    nodes = list(G.nodes())
    cmap = {
        node: sns.color_palette("colorblind")[9]  # light blue
        if (node.startswith("DB") and len(node) == 7)  # drugs
        else sns.color_palette("colorblind")[1]  # orange
        if node in targets  # drug targets and related genes
        else sns.color_palette("colorblind")[8]  # yellow
        for node in nodes  # related genes
    }
    colors = [mpl.colors.to_hex(cmap[node]) for node in nodes]
    net = Network("100%", "100%")
    net.add_nodes(nodes, label=nodes, color=colors)
    for head, tail in G.edges:
        net.add_edge(head, tail)
    net.set_options(
        """
        var options = {
          "edges": {
            "color": {
              "inherit": true
            },
            "font": {
              "size": 0
            },
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            }
          },
          "interaction": {
            "hideEdgesOnDrag": true
          },
          "physics": {
            "forceAtlas2Based": {
              "springLength": 100,
              "damping": 5,
              "avoidOverlap": 0.1
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "timestep": 0.25
          }
        }
        """
    )
    net.save_graph(
        f"data/results/{disease_name.replace(' ', '')}/drug_target_gene_network.html"
    )


def draw_gene_target_drug_sankey(disease_name, disease_genes, height=2000, width=1500):
    # commented parts are for adding a fourth column for IGSEA FDR values color-coded from blue to red (but misleading and ugly)
    import pandas as pd
    import numpy as np
    import re
    import networkx as nx
    import seaborn as sns
    import plotly.graph_objects as go

    from databases import DrugBank

    log = logging.getLogger("drug_target_gene:draw_gene_target_drug_sankey")

    drugbank = DrugBank()

    drugs = get_drugs(disease_name)

    drug2fdr = (
        drugs[["DrugBank_ID", "FDR"]]
        .drop_duplicates(subset=["DrugBank_ID"])
        .set_index("DrugBank_ID")["FDR"]
        .apply(lambda fdr: fdr if fdr.startswith("<") else "%.5f" % float(fdr))
        .to_dict()
    )

    drugbank_drug_targets = {
        k: list(ts)
        for k, ts in {
            d.id: {
                t.symbol
                for t in d.targets
                if t.type == "protein" and t.organism == "Humans" and t.symbol != None
            }
            for d in drugbank.database
        }.items()
        if ts
    }

    ppi = pd.read_csv(
        "data/sources/interactome_slim.tsv.gz", sep="\t", compression="gzip"
    )
    interactome = nx.from_pandas_edgelist(ppi, source="source", target="target")
    interactome = interactome.to_undirected()

    drugs = set(drugs.drop_duplicates(subset=["DrugBank_ID"])["DrugBank_ID"].values)
    drug2target = {
        (drug, target) for drug in drugs for target in drugbank_drug_targets[drug]
    }
    targets = {edge[1] for edge in drug2target}
    related_genes = {gene for gene in disease_genes}
    gene2target = {
        (source, target)
        for source in related_genes
        if source in interactome.nodes
        for target in interactome.neighbors(source)
        if target in targets and source != target
    }
    related_genes = {g for g in related_genes if g in {gt[0] for gt in gene2target}}
    targets = {t for t in targets if t in {gt[1] for gt in gene2target}}
    gene2target.update(
        set(
            zip(
                related_genes.intersection(targets), related_genes.intersection(targets)
            )
        )
    )  # self-link genes that are both disease-related-genes and targets
    drug2target = {
        (drug, target)
        for drug, target in drug2target
        if target in {gt[1] for gt in gene2target}
    }

    node_labels = (
        list(related_genes)
        + list(targets)
        + [
            f"<b>{drug}</b> ({(lambda name: name+', ' if not '(' in name else '')(drugbank.get(drug).name)}IGSEA FDR: {drug2fdr[drug]})"
            for drug in drugs
        ]
        # + [drug2fdr[drug] for drug in drugs]
    )
    related_genes_dict = {node: n for n, node in enumerate(related_genes)}
    targets_dict = {node: n + len(related_genes) for n, node in enumerate(targets)}
    drugs_dict = {
        node: n + len(related_genes) + len(targets) for n, node in enumerate(drugs)
    }
    # IGSEA_dict = {
    #     node: n + len(related_genes) + len(targets) + len(drugs)
    #     for n, node in enumerate(drugs)
    # }

    edges = (
        [
            (related_genes_dict[gene], targets_dict[target])
            for gene, target in gene2target
        ]
        + [
            (targets_dict[target], drugs_dict[drug])
            for drug, target in drug2target
            if target
        ]
        # + [(drugs_dict[drug], IGSEA_dict[drug]) for drug in drugs]
    )

    source_nodes, target_nodes = zip(*edges)
    target_values = (
        pd.DataFrame(gene2target, columns=["source", "target"])["target"]
        .value_counts()
        .to_dict()
    )
    targets_drugs = {
        t: v
        for t, v in pd.DataFrame(drug2target, columns=["source", "target"])["target"]
        .value_counts()
        .to_dict()
        .items()
    }
    target_values = {
        targets_dict[t]: (v / targets_drugs[t]) for t, v in target_values.items()
    }
    # IGSEA_values = {
    #     IGSEA_dict[drug]: np.sum(
    #         [target_values[targets_dict[target]] for target in group["Target"].values]
    #     )
    #     for drug, group in pd.DataFrame(
    #         drug2target, columns=["Drug", "Target"]
    #     ).groupby("Drug")
    # }
    # values = [
    #     target_values.get(edge[0], IGSEA_values.get(edge[1], 1)) for edge in edges
    # ]
    values = [target_values.get(edge[0], 1) for edge in edges]

    ATC_codes = {
        drug: {code.level1.code for code in drugbank.get(drug).atc_codes}
        for drug in drugs
    }
    all_ATC = sorted({code for codes in ATC_codes.values() for code in codes})
    ATC_cmap = dict(
        zip(all_ATC, sns.color_palette("Spectral_r", n_colors=len(all_ATC)))
    )
    ATC_colors = {
        drug: f"rgba{tuple(int(v*255) for v in ATC_cmap[next(iter(ATC_codes[drug]))])+(0.8,)}"
        if len(ATC_codes[drug]) == 1
        else f"rgba{tuple(int(v*255) for v in sns.color_palette('pastel')[-3])+(0.8,)}"
        if len(ATC_codes[drug]) == 0
        else f"rgba{tuple(int(v*255) for v in sns.color_palette('pastel')[4])+(0.8,)}"
        for drug in drugs
    }
    drugs_color = {
        drug: f"rgba{tuple(int(v*255) for v in ATC_cmap[next(iter(ATC_codes[drug]))])+(0.8,)}"
        if len(ATC_codes[drug]) == 1
        else f"rgba{tuple(int(v*255) for v in sns.color_palette('pastel')[-3])+(0.8,)}"
        if len(ATC_codes[drug]) == 0
        else f"rgba{tuple(int(v*255) for v in sns.color_palette('pastel')[4])+(0.8,)}"
        for drug in drugs
    }
    drug_nodes_color = {
        drugs_dict[drug]: color.replace("0.8", "0.25")
        for drug, color in drugs_color.items()
    }
    targets_color = {
        target: "rgba("
        + ", ".join(
            np.average(
                np.array(
                    [
                        re.findall(
                            "rgba\(([0-9]{1,3}), ([0-9]{1,3}), ([0-9]{1,3}), 0.[0-9]{1}\)",
                            ATC_colors[drug],
                        )[0]
                        for drug in group["Drug"]
                    ]
                ).astype(int),
                axis=0,
            )
            .astype(int)
            .astype(str)
            .tolist()
        )
        + ", 0.8)"
        for target, group in pd.DataFrame(
            drug2target, columns=["Drug", "Target"]
        ).groupby("Target")
    }
    target_nodes_color = {
        targets_dict[target]: color.replace("0.8", "0.15")
        for target, color in targets_color.items()
    }
    related_genes_color = {
        gene: "rgba("
        + ", ".join(
            np.average(
                np.array(
                    [
                        re.findall(
                            "rgba\(([0-9]{1,3}), ([0-9]{1,3}), ([0-9]{1,3}), 0.[0-9]{1}\)",
                            targets_color[target],
                        )[0]
                        for target in group["Target"]
                    ]
                ).astype(int),
                axis=0,
            )
            .astype(int)
            .astype(str)
            .tolist()
        )
        + ", 0.8)"
        for gene, group in pd.DataFrame(
            gene2target, columns=["Gene", "Target"]
        ).groupby("Gene")
    }
    related_gene_nodes_color = {
        related_genes_dict[rg]: color.replace("0.8", "0.2")
        for rg, color in related_genes_color.items()
    }
    # IGSEA_color = {
    #     drug: "rgba"
    #     + str(
    #         sns.color_palette("coolwarm", as_cmap=True)(
    #             (
    #                 lambda fdr: 0
    #                 if fdr.startswith("<")
    #                 else int((float(fdr) / 0.25) * 255)
    #             )(fdr),
    #             alpha=0.8,
    #         )
    #     )
    #     for drug, fdr in drug2fdr.items()
    # }
    # def save_igsea_fdr_colorbar():
    #     import matplotlib.pyplot as plt
    #     import matplotlib as mpl
    #
    #     fig = plt.figure()
    #     ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
    #     ax.set_title("IGSEA FDR")
    #     cbar = mpl.colorbar.ColorbarBase(ax, orientation="horizontal",
    #                                    cmap=sns.color_palette("coolwarm", as_cmap=True),
    #                                    norm=mpl.colors.Normalize(0, 0.25),
    #                                    ticks=[0, 0.25])
    #
    #     plt.savefig("tmp/IGSEA_FDR_colorbar.svg", bbox_inches="tight")
    #     plt.close()
    # save_igsea_fdr_colorbar()

    sankey = go.Figure(
        data=[
            go.Sankey(
                node={
                    "label": node_labels,
                    "color": [related_genes_color[gene] for gene in related_genes]
                    + [targets_color[target] for target in targets]
                    + [drugs_color[drug] for drug in drugs],
                    # + [IGSEA_color[drug] for drug in drugs],
                    "thickness": 40,
                },
                link={
                    "source": source_nodes,
                    "target": target_nodes,
                    "value": values,
                    "color": [
                        target_nodes_color[tn]
                        if tn < (len(related_genes) + len(targets))
                        else drug_nodes_color[tn]
                        if tn < (len(related_genes) + len(targets) + len(drugs))
                        else "rgba(0,0,0,0.015)"  # transparent edges between drugs and IGSEA FDR
                        for tn in target_nodes
                    ],
                },
            )
        ]
    )

    sankey.update_layout(font_size=height * 0.8 / len(related_genes))
    sankey.write_image(
        file=f"data/results/{disease_name.replace(' ', '')}/gene_target_drug_sankey.svg",
        format="svg",
        width=width,
        height=height,
    )


if __name__ == "__main__":
    # from data.sources.Alzheimer.get_alzheimer_related_genes import alzheimer_genes
    from data.sources.Huntington.get_huntington_related_genes import huntington_genes
    from data.sources.MultipleSclerosis.get_multiple_sclerosis_related_genes import (
        multiple_sclerosis_genes,
    )

    for disease_name, disease_genes in (
        # ("Alzheimer", alzheimer_genes.keys()),
        ("Huntington", huntington_genes.keys()),
        ("Multiple Sclerosis", multiple_sclerosis_genes.keys()),
    ):
        log.info(f"Building Drug-Target-RelatedGene Networks for {disease_name}")

        draw_drug_target_gene_network(disease_name, disease_genes)
        draw_gene_target_drug_sankey(disease_name, disease_genes)
