import pandas as pd
import numpy as np

import fastobo
from tqdm import tqdm
import os
import re
import requests
import inflection

from utils import Singleton, Database, get_best_match, profile, camelize

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "databases"
log = logging.getLogger(logging_name)


################################################################################
#
#                                   Databases
#
################################################################################


# @profile()
class NCBI(Database, metaclass=Singleton):
    """
    NCBI reference class
    Contains info about genes
    For mapping all protein names and aliases to the common nomenclature of official symbols of the NCBI gene database
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="Public Domain",
            license_url="https://www.ncbi.nlm.nih.gov/home/about/policies/#copyright",
        )
        self.__geneinfo = self._add_file(
            url="https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
            update=update,
        )
        self.__geneinfo.content = self.__geneinfo.content.query(
            "Full_name_from_nomenclature_authority != '-'"
        ).drop_duplicates(
            subset="Symbol_from_nomenclature_authority", keep=False, ignore_index=True,
        )  # keep only unique official symbols
        # NCBI official symbols
        self.__ncbi_symbols = tuple(
            self.__geneinfo()["Symbol_from_nomenclature_authority"]
        )
        # NCBI official ids
        self.__ncbi_ids = tuple(self.__geneinfo()["GeneID"].astype({"GeneID": int}))

        # NCBI unique synonyms (not ambiguous)
        self.__unique_synonyms = {}
        shared_synonyms = {}
        for symbol, synonyms, ncbi_symbol in self.__geneinfo()[
            ["Symbol_from_nomenclature_authority", "Synonyms", "Symbol"]
        ].itertuples(index=False):
            if synonyms != "-":
                for synonym in synonyms.split("|") + [ncbi_symbol]:
                    # ambiguity check
                    if synonym in shared_synonyms:
                        shared_synonyms[synonym].append(symbol)
                    elif synonym in self.__unique_synonyms:
                        shared_synonyms[synonym] = [
                            self.__unique_synonyms[synonym],
                            symbol,
                        ]
                        del self.__unique_synonyms[synonym]
                    else:
                        self.__unique_synonyms[synonym] = symbol

        self.__gene2name = self.__geneinfo()[
            [
                "Symbol_from_nomenclature_authority",
                "Full_name_from_nomenclature_authority",
            ]
        ].rename(
            columns={
                "Symbol_from_nomenclature_authority": "geneSymbol",
                "Full_name_from_nomenclature_authority": "geneName",
            }
        )
        self.__gene2name_asdict = self.__gene2name.set_index("geneSymbol")[
            "geneName"
        ].to_dict()
        self.__id2symbol = (
            self.__geneinfo()[["GeneID", "Symbol_from_nomenclature_authority"]]
            .rename(
                columns={
                    "GeneID": "geneId",
                    "Symbol_from_nomenclature_authority": "geneSymbol",
                }
            )
            .astype({"geneId": int})
        )
        self.__id2symbol_asdict = self.__id2symbol.set_index("geneId")[
            "geneSymbol"
        ].to_dict()

        self.__symbol2id = (
            self.__geneinfo()[["Symbol_from_nomenclature_authority", "GeneID"]]
            .rename(
                columns={
                    "Symbol_from_nomenclature_authority": "geneSymbol",
                    "GeneID": "geneId",
                }
            )
            .astype({"geneId": int})
        )
        self.__symbol2id_asdict = self.__symbol2id.set_index("geneSymbol")[
            "geneId"
        ].to_dict()

        self.__omim2ncbi = set()
        self.__hgnc2ncbi = set()
        self.__ensembl2ncbi = set()
        for symbol, xref in self.__geneinfo()[
            ["Symbol_from_nomenclature_authority", "dbXrefs"]
        ].itertuples(index=False):
            if xref != "-":
                for key, value in {
                    id.split(":", 1)[0]: id.split(":", 1)[1] for id in xref.split("|")
                }.items():
                    if key == "MIM":
                        self.__omim2ncbi.add((value, symbol))
                    elif key == "HGNC":
                        self.__hgnc2ncbi.add((value, symbol))
                    elif key == "Ensembl":
                        self.__ensembl2ncbi.add((value, symbol))
        self.__omim2ncbi = pd.DataFrame(self.__omim2ncbi, columns=["OMIM", "NCBI"])
        self.__omim2ncbi = self.__omim2ncbi[
            ~(
                self.__omim2ncbi["OMIM"].duplicated(keep=False)
                + self.__omim2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__omim2ncbi_asdict = self.__omim2ncbi.set_index("OMIM")["NCBI"].to_dict()
        self.__hgnc2ncbi = pd.DataFrame(self.__hgnc2ncbi, columns=["HGNC", "NCBI"])
        self.__hgnc2ncbi = self.__hgnc2ncbi[
            ~(
                self.__hgnc2ncbi["HGNC"].duplicated(keep=False)
                + self.__hgnc2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__hgnc2ncbi_asdict = self.__hgnc2ncbi.set_index("HGNC")["NCBI"].to_dict()
        self.__ensembl2ncbi = pd.DataFrame(
            self.__ensembl2ncbi, columns=["Ensembl", "NCBI"]
        )
        self.__ensembl2ncbi = self.__ensembl2ncbi[
            ~(
                self.__ensembl2ncbi["Ensembl"].duplicated(keep=False)
                + self.__ensembl2ncbi["NCBI"].duplicated(keep=False)
            )
        ]
        self.__ensembl2ncbi_asdict = self.__ensembl2ncbi.set_index("Ensembl")[
            "NCBI"
        ].to_dict()

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def symbols(self):
        """returns NCBI official symbols"""
        return self.__ncbi_symbols

    @property
    def database(self):
        return self.__geneinfo().copy()

    def check_symbol(self, symbol=None, aliases=[]):
        """
        Checks if symbols or aliases are ambiguous or can be mapped to unique NCBI symbol
        Returns the official NCBI symbol if there is one, False otherwise
        """
        if symbol is pd.NA or (not symbol and not aliases):
            return None
        elif symbol in self.__ncbi_symbols:
            return symbol
        elif symbol in self.__unique_synonyms:
            return self.__unique_synonyms[symbol]
        else:
            for alias in aliases:
                if alias in self.__ncbi_symbols:
                    return alias
                elif alias in self.__unique_synonyms:
                    return self.__unique_synonyms[alias]
        return None

    def get_symbol_by_id(self, id):
        """
        Checks if id in official NCBI ids
        Returns the official corresponding NCBI symbol if there is one, None otherwise
        """
        if not isinstance(id, int):
            if isinstance(id, str) and ("NCBI:" in id or "NCBIGene:" in id):
                id = id.split(":")[-1]
            try:
                id = int(id)
            except:
                return None
        if not id:
            return None
        elif id in self.__ncbi_ids:
            return self.check_symbol(self.__id2symbol_asdict.get(id))
        else:
            return None

    def get_id_by_symbol(self, symbol):
        """
        Checks if symbol in official NCBI symbols
        Returns the official corresponding NCBI id if there is one, None otherwise
        """
        if isinstance(symbol, str) and ("NCBI:" in symbol or "NCBIGene:" in symbol):
            symbol = symbol.split(":")[-1]
        if not symbol:
            return None
        elif symbol in self.__ncbi_symbols:
            return int(self.__symbol2id_asdict.get(self.check_symbol(symbol)))
        else:
            return None

    @property
    def id2symbol(self):
        return self.__id2symbol.copy()

    @property
    def symbol2id(self):
        return self.__symbol2id.copy()

    def get_name(self, symbol):
        """
        Given an official symbol returns the Full_name_from_nomenclature_authority
        """
        return self.__gene2name_asdict.get(symbol, f"{symbol} Not Found")

    @property
    def gene2name(self):
        return self.__gene2name.copy()

    @property
    def omim2ncbi(self):
        return self.__omim2ncbi.copy()

    @property
    def omim2ncbi_asdict(self):
        return self.__omim2ncbi_asdict.copy()

    def get_symbol_by_omim(self, omim):
        """
        Given a OMIM id returns the NCBI gene symbol
        """
        return self.__omim2ncbi_asdict.get(omim, f"{omim} Not Found")

    @property
    def hgnc2ncbi(self):
        return self.__hgnc2ncbi.copy()

    @property
    def hgnc2ncbi_asdict(self):
        return self.__hgnc2ncbi_asdict.copy()

    def get_symbol_by_hgnc(self, hgnc):
        """
        Given a HGNC id returns the NCBI gene symbol
        """
        return self.__hgnc2ncbi_asdict.get(hgnc, f"{hgnc} Not Found")

    @property
    def ensembl2ncbi(self):
        return self.__ensembl2ncbi.copy()

    @property
    def ensembl2ncbi_asdict(self):
        return self.__ensembl2ncbi_asdict.copy()

    def get_symbol_by_ensembl(self, ensembl):
        """
        Given a Ensembl id returns the NCBI gene symbol
        """
        return self.__ensembl2ncbi_asdict.get(ensembl, f"{ensembl} Not Found")


# @profile()
class HGNC(Database, metaclass=Singleton):
    """
    HGNC reference class
    Contains info about genes
    For mapping all protein names, aliases and symbols to the approved human gene nomenclature
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="Freely Available",
            license_url="https://www.genenames.org/about/",
        )
        self.__db = self._add_file(
            url="http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/locus_types/gene_with_protein_product.txt",
            update=update,
        )
        # HGNC official symbols
        self.__hgnc_symbols = tuple(self.__db()["symbol"])
        # HGNC official ids
        self.__hgnc_ids = tuple(self.__db()["hgnc_id"])

        # HGNC unique synonyms (not ambiguous)
        self.__unique_synonyms = {}
        shared_synonyms = {}
        for symbol, synonyms in (
            self.__db()[["symbol", "alias_symbol"]]
            .dropna(subset=["alias_symbol"])
            .itertuples(index=False)
        ):
            for synonym in synonyms.split("|"):
                # ambiguity check
                if synonym in shared_synonyms:
                    shared_synonyms[synonym].append(symbol)
                elif synonym in self.__unique_synonyms:
                    shared_synonyms[synonym] = [
                        self.__unique_synonyms[synonym],
                        symbol,
                    ]
                    del self.__unique_synonyms[synonym]
                else:
                    self.__unique_synonyms[synonym] = symbol

        self.__gene2name = self.__db()[["symbol", "name"]].rename(
            columns={"symbol": "geneSymbol", "name": "geneName",}
        )
        self.__gene2name_asdict = self.__gene2name.set_index("geneSymbol")[
            "geneName"
        ].to_dict()
        self.__id2symbol = self.__db()[["hgnc_id", "symbol"]].rename(
            columns={"hgnc_id": "geneId", "symbol": "geneSymbol"}
        )
        self.__id2symbol_asdict = self.__id2symbol.set_index("geneId")[
            "geneSymbol"
        ].to_dict()

        self.__symbol2id = self.__db()[["symbol", "hgnc_id"]].rename(
            columns={"symbol": "geneSymbol", "hgnc_id": "geneId"}
        )
        self.__symbol2id_asdict = self.__symbol2id.set_index("geneSymbol")[
            "geneId"
        ].to_dict()

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def symbols(self):
        """returns HGNC official symbols"""
        return self.__hgnc_symbols

    @property
    def database(self):
        return self.__db().copy()

    def check_symbol(self, symbol, aliases=[]):
        """
        Checks if symbols or aliases are ambiguous or can be mapped to unique HGNC symbol
        Returns the official HGNC symbol if there is one, False otherwise
        """
        if not symbol:
            return None
        elif symbol in self.__hgnc_symbols:
            return symbol
        elif symbol in self.__unique_synonyms:
            return self.__unique_synonyms[symbol]
        else:
            for alias in aliases:
                if alias in self.__hgnc_symbols:
                    return alias
                elif alias in self.__unique_synonyms:
                    return self.__unique_synonyms[alias]
        return None

    def get_symbol_by_id(self, id):
        """
        Checks if id in official HGNC ids and returns the official corresponding HGNC symbol if there is one, None otherwise
        """
        if not id:
            return None
        if not isinstance(id, str):
            id = str(id)
        if not id.startswith("HGNC:"):
            id = "HGNC:" + id
        if not id:
            return None
        elif id in self.__hgnc_ids:
            return self.check_symbol(self.__id2symbol_asdict.get(id))
        else:
            return None

    def get_id_by_symbol(self, symbol):
        """
        Checks if symbol in official HGNC symbols
        Returns the official corresponding HGNC id if there is one, None otherwise
        """
        if not symbol:
            return None
        elif symbol in self.__hgnc_symbols:
            return self.__symbol2id_asdict.get(self.check_symbol(symbol))
        else:
            return None

    @property
    def id2symbol(self):
        return self.__id2symbol.copy()

    def get_name(self, symbol):
        """
        Given an official symbol returns the gene name
        """
        return self.__gene2name_asdict.get(symbol, f"{symbol} Not Found")

    @property
    def gene2name(self):
        return self.__gene2name.copy()


# @profile()
class GO(Database, metaclass=Singleton):
    """
    Gene Ontology reference class
    Contains info about biological processes,
    molecular functions,
    and cellular components
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="http://geneontology.org/docs/go-citation-policy/#license",
            requirements=[NCBI],
        )

        goa_columns = [
            "DB",
            "DB Object ID",
            "DB Object Symbol",
            "Qualifier",
            "GO ID",
            "DB Reference",
            "Evidence Code",
            "With (or) From",
            "Aspect",
            "DB Object Name",
            "DB Object Synonym",
            "DB Object Type",
            "Taxon",
            "Date",
            "Assigned By",
            "Annotatoin Extension",
            "Gene Product From ID",
        ]
        self.__gene_associations = self._add_file(
            url="http://geneontology.org/gene-associations/goa_human.gaf.gz",
            skiprows=41,
            names=goa_columns,
            usecols=[
                "DB Object Symbol",
                "Qualifier",
                "GO ID",
                "Evidence Code",
                "DB Object Synonym",
            ],
        )
        self.__gene_associations.content = self.__gene_associations.content[
            self.__gene_associations.content["Evidence Code"].isin(
                [
                    "EXP",
                    "IDA",
                    "IPI",
                    "IMP",
                    "IGI",
                    "IEP",
                    "HTP",
                    "HDA",
                    "HMP",
                    "HGI",
                    "HEP",
                    "IBA",
                    "IBD",
                    "IKR",
                    "IRD",
                    "TAS",
                    "IC",
                    "NAS",
                ]
            )
        ].reset_index(drop=True)
        self.__gene2go = set()
        for symbol, qualifier, goid, synonyms in tqdm(
            self.__gene_associations()[
                ["DB Object Symbol", "Qualifier", "GO ID", "DB Object Synonym"]
            ].itertuples(index=False),
            desc=f"Collecting data from {self.__gene_associations.filename}",
        ):
            if isinstance(synonyms, str):  # in order to avoid nans
                symbol = self.__NCBI.check_symbol(symbol, synonyms.split("|"))
            else:
                symbol = self.__NCBI.check_symbol(symbol)
            if symbol:
                self.__gene2go.add((symbol, qualifier, goid))
        self.__gene2go = pd.DataFrame(
            self.__gene2go, columns=["geneSymbol", "relation", "goTerm"]
        )

        def fix_go_obofile(filepath):
            """
            Tries to fix the go go.obo file to be readable by fastobo
            """
            content = []
            with open(filepath, "r") as infile:
                for line in infile:
                    if line.split(":")[0] in ["synonym", "def"]:
                        try:
                            s = re.findall(
                                '[a-zA-Z]+:\ ".+"\ [A-Z \s _]*\[(.+)\]|$', line.strip()
                            )[0]
                            if len(s):
                                line = line.replace(s, s.replace(" ", "_"))
                        except:
                            pass
                    elif (
                        line.split(":")[0] == "xref" and line.count(":") == 1
                    ):  # drop xref lines with non-standard format
                        line = ""
                    elif (
                        line.split(":")[0] == "xref" and line.count(":") > 1
                    ):  # removes space between database code and relative id in xref
                        line = f'xref: {line.split("xref: ")[-1].replace(": ", ":")}'
                    content.append(line)
            filepath = os.path.join(
                os.path.dirname(filepath), f"cleaned-{os.path.basename(filepath)}"
            )
            with open(filepath, "w") as outfile:
                for line in content:
                    outfile.write(line)
            return filepath

        self.__db = self._add_file(
            url="http://geneontology.org/ontology/go-basic.obo",
            fix_function=fix_go_obofile,
        )
        self.__go2name = set()
        self.__goRelationship = set()
        self.__go2namespace = set()
        for frame in tqdm(
            self.__db(), desc=f"Collecting info from {self.__db.filename}"
        ):
            is_obsolete = False
            relationship_tmp = []
            for clause in frame:
                if isinstance(clause, fastobo.term.IsObsoleteClause):
                    is_obsolete = True
                    break
                elif isinstance(clause, fastobo.term.NameClause):
                    self.__go2name.add((str(frame.id), clause.name))
                elif isinstance(clause, fastobo.term.RelationshipClause):
                    relationship_tmp.append(
                        (str(frame.id), clause.typedef.unescaped, str(clause.term))
                    )
                elif isinstance(clause, fastobo.term.IsAClause):
                    relationship_tmp.append((str(frame.id), "is_a", str(clause.term)))
                elif isinstance(clause, fastobo.term.NamespaceClause):
                    self.__go2namespace.add(
                        (str(frame.id), camelize(clause.namespace.escaped))
                    )
            if not is_obsolete:
                self.__goRelationship.update(relationship_tmp)

        self.__gene2go = self.__gene2go.merge(
            pd.DataFrame(self.__go2name, columns=["goTerm", "goName"]),
            on="goTerm",
            how="left",
        )
        self.__gene2go = (
            self.__gene2go.merge(self.__NCBI.gene2name, on="geneSymbol")
            .dropna(subset=["geneName"])
            .reset_index(drop=True)
        )
        self.__gene2go = (
            self.__gene2go.merge(
                pd.DataFrame(self.__go2namespace, columns=["goTerm", "goNamespace"]),
                on="goTerm",
                how="left",
            )
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "geneSymbol": "string",
                    "relation": "category",
                    "goTerm": "string",
                    "goName": "string",
                    "geneName": "string",
                    "goNamespace": "category",
                }
            )
        )

        # keep local (only needed for building the ontology)
        gene2go = self.__gene2go.rename(
            columns={
                "geneSymbol": "subject",
                "goTerm": "object",
                "geneName": "subjectName",
                "goName": "objectName",
            }
        )
        gene2go["subjectType"] = ["protein"] * len(gene2go)
        gene2go.insert(len(gene2go.columns), "source", self.__class__.__name__)
        gene2go = (
            gene2go.merge(
                pd.DataFrame(self.__go2namespace, columns=["object", "objectType"]),
                on="object",
                how="left",
            )
            .dropna(subset=["objectType"])
            .reset_index(drop=True)
        )
        gene2go = gene2go[
            [
                "subject",
                "relation",
                "object",
                "subjectName",
                "objectName",
                "subjectType",
                "objectType",
                "source",
            ]
        ].astype(
            {
                "subject": "string",
                "relation": "category",
                "object": "string",
                "subjectName": "string",
                "objectName": "string",
                "subjectType": "category",
                "objectType": "category",
                "source": "string",
            }
        )

        self.__ontology = pd.DataFrame(
            self.__goRelationship, columns=["subject", "relation", "object"]
        )
        for column in ["subject", "object"]:
            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__go2name, columns=[column, f"{column}Name"]),
                on=column,
                how="left",
            )
            self.__ontology = self.__ontology.merge(
                pd.DataFrame(self.__go2namespace, columns=[column, f"{column}Type"]),
                on=column,
                how="left",
            )
        self.__ontology.insert(
            len(self.__ontology.columns), "source", self.__class__.__name__
        )
        self.__ontology = (
            self.__ontology[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )
        self.__ontology = self.__ontology.append(gene2go).drop_duplicates(
            ignore_index=True
        )

        # for every namespace set a property to retrieve ids and names
        namespaces = {t[1] for t in self.__go2namespace}
        for namespace in namespaces:
            value = (
                self.__ontology[self.__ontology["subjectType"] == namespace][
                    ["subject", "subjectName"]
                ]
                .rename(
                    columns={"subject": namespace, "subjectName": f"{namespace}Name"}
                )
                .append(
                    self.__ontology[self.__ontology["objectType"] == namespace][
                        ["object", "objectName"]
                    ].rename(
                        columns={"object": namespace, "objectName": f"{namespace}Name"}
                    )
                )
            )
            setattr(
                self.__class__,
                inflection.pluralize(namespace),
                property(lambda x: value.copy()),
            )  # uselessy too complex? maybe

        # dicts for info retrieval
        self.__go2name_asdict = dict(self.__go2name)
        self.__go2namespace_asdict = dict(self.__go2namespace)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def gene2go(self):
        return self.__gene2go.copy()

    @property
    def go2name(self):
        return self.__go2name.copy()

    @property
    def go2namespace(self):
        return self.__go2namespace.copy()

    @property
    def ontology(self):
        return self.__ontology.copy()

    def get_name(self, id):
        """Returns the name of a GO id"""
        if not id.startswith("GO:"):
            id = f"GO:{id}"
        return self.__go2name_asdict.get(id, f"Id {id} not found")

    def get_namespace(self, id):
        """Returns the namespace associated to a GO id"""
        if not id.startswith("GO:"):
            id = f"GO:{id}"
        return self.__go2namespace_asdict.get(id, f"Id {id} not found")


# @profile()
class DisGeNET(Database, metaclass=Singleton):
    """
    DisGeNET reference class
    Contains info about gene-disease associations
    """

    def __init__(self, update=None):
        try:
            email = os.environ["DISGENET_EMAIL"]
            password = os.environ["DISGENET_PASSWORD"]
        except KeyError:
            try:
                from dotenv import dotenv_values

                credentials = dotenv_values()
                email = credentials["DISGENET_EMAIL"]
                password = credentials["DISGENET_PASSWORD"]
            except:
                raise RuntimeError(
                    "No DisGeNET credentials found, "
                    "gene-disease associations will not be collected"
                )
        auth_url = "https://www.disgenet.org/api/auth/"
        response = requests.post(auth_url, data={"email": email, "password": password})
        token = response.json()["token"]
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})
        Database.__init__(
            self,
            update=update,
            license="CC BY-NC-SA 4.0",
            license_url="https://www.disgenet.org/legal",
            registration_required=True,
            session=session,
            requirements=[NCBI],
        )
        self.__disease_mapping = self._add_file(
            url="https://www.disgenet.org/static/disgenet_ap1/files/downloads/disease_mappings.tsv.gz",
            usecols=["diseaseId", "vocabulary", "code"],
        )
        self.__umls2mondo = {
            (umls, f"MONDO:{mondo}")
            for umls, mondo in self.__disease_mapping()
            .query("vocabulary == 'MONDO'")[["diseaseId", "code"]]
            .itertuples(index=False)
        }
        self.__db = self._add_file(
            url="https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz",
            dtype={
                "geneId": int,
                "geneSymbol": "string",
                "DSI": float,
                "DPI": float,
                "diseaseId": "string",
                "diseaseName": "string",
                "diseaseType": "string",
                "diseaseClass": "string",
                "diseaseSemanticType": "string",
                "score": float,
                "EI": float,
                "YearInitial": "string",
                "YearFinal": "string",
                "NofPmids": int,
                "NofSnps": int,
                "source": "string",
            },
        )
        self.__db.content["geneSymbol"] = self.__db.content["geneSymbol"].map(
            self.__NCBI.check_symbol
        )
        self.__db.content.dropna(subset=["geneSymbol"], inplace=True)
        self.__db.content.drop_duplicates(ignore_index=True, inplace=True)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def umls2mondo(self):
        return self.__umls2mondo.copy()

    @property
    def database(self):
        return self.__db().copy()

    @property
    def disease_mapping(self):
        return self.__disease_mapping().copy()


# @profile()
class DrugBank(Database, metaclass=Singleton):
    """
    DrugBank reference class
    Contains info about drugs
    """

    def __init__(self, update=None):
        try:
            email = os.environ["DRUGBANK_EMAIL"]
            password = os.environ["DRUGBANK_PASSWORD"]
        except KeyError:
            try:
                from dotenv import dotenv_values

                credentials = dotenv_values()
                email = credentials["DRUGBANK_EMAIL"]
                password = credentials["DRUGBANK_PASSWORD"]
            except:
                raise RuntimeError(
                    "No DrugBank credentials found, "
                    "drugs information will not be collected"
                )
        session = requests.Session()
        import base64

        session.headers.update(
            {
                "Authorization": f"Basic {base64.b64encode(f'{email}:{password}'.encode('ascii')).decode('ascii')}"
            }
        )
        Database.__init__(
            self,
            update=update,
            license="CC BY-NC 4.0",
            license_url="https://go.drugbank.com/releases/latest#full",
            registration_required=True,
            session=session,
            requirements=[NCBI],
        )

        def read_drugbank_full_database(filepath):
            # preparing namedtuples
            from collections import namedtuple

            atc_code = namedtuple(
                typename="atc_code",
                field_names=["level5", "level4", "level3", "level2", "level1"],
            )
            atc_code_level = namedtuple(
                typename="atc_code_level", field_names=["code", "name"]
            )
            category = namedtuple(typename="category", field_names=["name", "mesh_id"])
            interacting_drug = namedtuple(
                typename="interacting_drug",
                field_names=["name", "drugbank_id", "description"],
            )
            experimental_property = namedtuple(
                typename="experimental_property",
                field_names=["name", "value", "source"],
            )
            calculated_property = namedtuple(
                typename="calculated_property", field_names=["name", "value", "source"]
            )
            external_identifiers = namedtuple(
                typename="external_identifiers", field_names=["resource", "id"]
            )
            pathway = namedtuple(
                typename="pathway", field_names=["name", "smpdb_id", "category"]
            )
            protein_function = namedtuple(
                typename="protein_function", field_names=["general", "specific"]
            )
            protein = namedtuple(
                typename="protein",
                field_names=[
                    "drug_actions",
                    "cellular_location",
                    "chromosome_location",
                    "function",
                    "id",
                    "name",
                    "organism",
                    "swiss_prot_id",
                    "symbol",
                    "synonyms",
                    "type",
                ],
            )
            biological_entity = namedtuple(
                typename="biological_entity",
                field_names=["drug_actions", "id", "name", "organism", "type"],
            )
            small_molecule = namedtuple(
                typename="small_molecule",
                field_names=[
                    "affected_organisms",
                    "atc_codes",
                    "calculated_properties",
                    "carriers",
                    "cas_number",
                    "categories",
                    "combined_ingredients",  # ingredients of approved mixture products
                    "description",
                    "drug_interactions",
                    "enzymes",
                    "experimental_properties",
                    "external_identifiers",
                    "groups",
                    "id",
                    "indication",
                    "mechanism_of_action",
                    "name",
                    "pathways",
                    "pharmacodynamics",
                    "targets",
                    "toxicity",
                    "transporters",
                    "type",
                ],
            )

            biotech = namedtuple(
                typename="biotech",
                field_names=[
                    "affected_organisms",
                    "atc_codes",
                    "carriers",
                    "cas_number",
                    "categories",
                    "combined_ingredients",  # ingredients of approved mixture products
                    "description",
                    "drug_interactions",
                    "enzymes",
                    "experimental_properties",
                    "external_identifiers",
                    "groups",
                    "id",
                    "indication",
                    "mechanism_of_action",
                    "name",
                    "pathways",
                    "pharmacodynamics",
                    "targets",
                    "toxicity",
                    "transporters",
                    "type",
                ],
            )

            # parsing database
            import zipfile

            with zipfile.ZipFile(filepath) as z:
                with z.open(z.filelist[0].filename) as f:
                    from lxml import objectify

                    drugbank_database = objectify.parse(f).getroot().drug
            drugs_namedtuple = namedtuple(
                typename="drugs",
                field_names=tuple(d["drugbank-id"] for d in drugbank_database),
            )
            drugs = drugs_namedtuple(
                *[
                    small_molecule(
                        tuple(
                            organism.text
                            for organism in d["affected-organisms"].getchildren()
                        ),
                        tuple(
                            atc_code(
                                *[
                                    atc_code_level(
                                        codes.values()[0], f"Substance level: {d.name}"
                                    )
                                ]
                                + [
                                    atc_code_level(code.values()[0], code.text)
                                    for code in codes.iterchildren()
                                ]
                            )
                            for codes in d["atc-codes"].iterchildren()
                        ),
                        tuple(
                            calculated_property(
                                str(prop.kind), str(prop.value), str(prop.source)
                            )
                            for prop in d["calculated-properties"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in carrier.actions.getchildren()
                                ),
                                str(carrier.polypeptide["cellular-location"]),
                                str(carrier.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(carrier.polypeptide["general-function"]),
                                    str(carrier.polypeptide["specific-function"]),
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                str(carrier.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(carrier.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in carrier.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(carrier, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in carrier.actions
                                    if hasattr(action, "action")
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                "biological_entity",
                            )
                            for carrier in d.carriers.getchildren()
                        ),
                        str(d["cas-number"]),
                        tuple(
                            category(str(cat.category), str(cat["mesh-id"]))
                            for cat in d.categories.iterchildren()
                        ),
                        tuple(
                            sorted(
                                {
                                    ingredient
                                    for mixture in d.mixtures.iterchildren()
                                    for ingredient in str(mixture.ingredients).split(
                                        " + "
                                    )
                                    if ingredient != d.name
                                }
                            )
                        ),
                        str(d.description),
                        tuple(
                            interacting_drug(
                                str(interaction.name),
                                str(interaction["drugbank-id"]),
                                str(interaction.description),
                            )
                            for interaction in d["drug-interactions"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in enzyme.actions.getchildren()
                                ),
                                str(enzyme.polypeptide["cellular-location"]),
                                str(enzyme.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(enzyme.polypeptide["general-function"]),
                                    str(enzyme.polypeptide["specific-function"]),
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                str(enzyme.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(enzyme.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in enzyme.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(enzyme, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in enzyme.actions
                                    if hasattr(action, "action")
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                "biological_entity",
                            )
                            for enzyme in d.enzymes.getchildren()
                        ),
                        tuple(
                            experimental_property(
                                str(prop.kind), str(prop.value), str(prop.source)
                            )
                            for prop in d["experimental-properties"].getchildren()
                        ),
                        tuple(
                            external_identifiers(str(xid.resource), str(xid.identifier))
                            for xid in d["external-identifiers"].getchildren()
                        ),
                        tuple(str(group) for group in d.groups.getchildren()),
                        str(d["drugbank-id"]),
                        str(d["indication"]),
                        str(d["mechanism-of-action"]),
                        str(d.name),
                        tuple(
                            pathway(str(p.name), str(p["smpdb-id"]), str(p.category))
                            for p in d.pathways.getchildren()
                        ),
                        str(d.pharmacodynamics),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in target.actions.getchildren()
                                ),
                                str(target.polypeptide["cellular-location"]),
                                str(target.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(target.polypeptide["general-function"]),
                                    str(target.polypeptide["specific-function"]),
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                str(target.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(target.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in target.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(target, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in target.actions
                                    if hasattr(action, "action")
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                "biological_entity",
                            )
                            for target in d.targets.getchildren()
                        ),
                        str(d.toxicity),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in transporter.actions.getchildren()
                                ),
                                str(transporter.polypeptide["cellular-location"]),
                                str(transporter.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(transporter.polypeptide["general-function"]),
                                    str(transporter.polypeptide["specific-function"]),
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                str(transporter.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(transporter.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in transporter.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(transporter, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in transporter.actions
                                    if hasattr(action, "action")
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                "biological_entity",
                            )
                            for transporter in d.transporters.getchildren()
                        ),
                        "small_molecule",
                    )
                    if d.values()[0] == "small molecule"
                    else biotech(
                        tuple(
                            organism.text
                            for organism in d["affected-organisms"].getchildren()
                        ),
                        tuple(
                            atc_code(
                                *[
                                    atc_code_level(
                                        codes.values()[0], f"Substance level: {d.name}"
                                    )
                                ]
                                + [
                                    atc_code_level(code.values()[0], code.text)
                                    for code in codes.iterchildren()
                                ]
                            )
                            for codes in d["atc-codes"].iterchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in carrier.actions.getchildren()
                                ),
                                str(carrier.polypeptide["cellular-location"]),
                                str(carrier.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(carrier.polypeptide["general-function"]),
                                    str(carrier.polypeptide["specific-function"]),
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                str(carrier.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(carrier.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in carrier.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(carrier, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in carrier.actions
                                    if hasattr(action, "action")
                                ),
                                str(carrier.id),
                                str(carrier.name),
                                str(carrier.organism),
                                "biological_entity",
                            )
                            for carrier in d.carriers.getchildren()
                        ),
                        str(d["cas-number"]),
                        tuple(
                            category(str(cat.category), str(cat["mesh-id"]))
                            for cat in d.categories.iterchildren()
                        ),
                        tuple(
                            sorted(
                                {
                                    ingredient
                                    for mixture in d.mixtures.iterchildren()
                                    for ingredient in str(mixture.ingredients).split(
                                        " + "
                                    )
                                    if ingredient != d.name
                                }
                            )
                        ),
                        str(d.description),
                        tuple(
                            interacting_drug(
                                str(interaction.name),
                                str(interaction["drugbank-id"]),
                                str(interaction.description),
                            )
                            for interaction in d["drug-interactions"].getchildren()
                        ),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in enzyme.actions.getchildren()
                                ),
                                str(enzyme.polypeptide["cellular-location"]),
                                str(enzyme.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(enzyme.polypeptide["general-function"]),
                                    str(enzyme.polypeptide["specific-function"]),
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                str(enzyme.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(enzyme.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in enzyme.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(enzyme, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in enzyme.actions
                                    if hasattr(action, "action")
                                ),
                                str(enzyme.id),
                                str(enzyme.name),
                                str(enzyme.organism),
                                "biological_entity",
                            )
                            for enzyme in d.enzymes.getchildren()
                        ),
                        tuple(
                            experimental_property(
                                str(prop.kind), str(prop.value), str(prop.source)
                            )
                            for prop in d["experimental-properties"].getchildren()
                        ),
                        tuple(
                            external_identifiers(str(xid.resource), str(xid.identifier))
                            for xid in d["external-identifiers"].getchildren()
                        ),
                        tuple(str(group) for group in d.groups.getchildren()),
                        str(d["drugbank-id"]),
                        str(d["indication"]),
                        str(d["mechanism-of-action"]),
                        str(d.name),
                        tuple(
                            pathway(str(p.name), str(p["smpdb-id"]), str(p.category))
                            for p in d.pathways.getchildren()
                        ),
                        str(d.pharmacodynamics),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in target.actions.getchildren()
                                ),
                                str(target.polypeptide["cellular-location"]),
                                str(target.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(target.polypeptide["general-function"]),
                                    str(target.polypeptide["specific-function"]),
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                str(target.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(target.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in target.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(target, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in target.actions
                                    if hasattr(action, "action")
                                ),
                                str(target.id),
                                str(target.name),
                                str(target.organism),
                                "biological_entity",
                            )
                            for target in d.targets.getchildren()
                        ),
                        str(d.toxicity),
                        tuple(
                            protein(
                                tuple(
                                    str(action)
                                    for action in transporter.actions.getchildren()
                                ),
                                str(transporter.polypeptide["cellular-location"]),
                                str(transporter.polypeptide["chromosome-location"]),
                                protein_function(
                                    str(transporter.polypeptide["general-function"]),
                                    str(transporter.polypeptide["specific-function"]),
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                str(transporter.polypeptide.values()[0]),
                                self.__NCBI.check_symbol(
                                    str(transporter.polypeptide["gene-name"])
                                ),
                                tuple(
                                    str(syn)
                                    for syn in transporter.polypeptide.synonyms.getchildren()
                                ),
                                "protein",
                            )
                            if hasattr(transporter, "polypeptide")
                            else biological_entity(
                                tuple(
                                    str(action.action)
                                    for action in transporter.actions
                                    if hasattr(action, "action")
                                ),
                                str(transporter.id),
                                str(transporter.name),
                                str(transporter.organism),
                                "biological_entity",
                            )
                            for transporter in d.transporters.getchildren()
                        ),
                        "biotech",
                    )
                    for d in tqdm(
                        drugbank_database, desc="Collecting Drugs Data from DrugBank"
                    )
                ]
            )

            return drugs

        self.__db = self._add_file(
            url=self.get_current_release_url(),
            custom_read_function=read_drugbank_full_database,
            retrieved_version=self.v,  # retrieved by get_current_release_url
        )

        self.__name2id = {drug.name: drug.id for drug in self.__db()}
        self.__drugNames = {drug.name for drug in self.__db()}

        self.__inchiKeyBase2name = {
            next(
                (
                    prop.value.split("-")[0]
                    for prop in d.calculated_properties
                    if prop.name == "InChIKey"
                ),
                None,
            ): d.name
            for d in self.__db()
            if d.type == "small_molecule"
        }
        del [self.__inchiKeyBase2name[None]]
        self.__inchiKeyBase2id = {
            next(
                (
                    prop.value.split("-")[0]
                    for prop in d.calculated_properties
                    if prop.name == "InChIKey"
                ),
                None,
            ): d.id
            for d in self.__db()
            if d.type == "small_molecule"
        }
        del [self.__inchiKeyBase2id[None]]

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db()

    @property
    def drugs(self):
        return self.__db()

    @property
    def inchiKeyBase2id(self):
        return self.__inchiKeyBase2id

    def get_id_by_inchiKeyBase(self, inchiKeyBase):
        return self.__inchiKeyBase2id.get(inchiKeyBase)

    @property
    def inchiKeyBase2name(self):
        return self.__inchiKeyBase2name

    def get_name_by_inchiKeyBase(self, inchiKeyBase):
        return self.__inchiKeyBase2name.get(inchiKeyBase)

    def get(self, query):
        """
            Returns a namedtuple with relevant data about the requested drug

            Accepts DrugBank IDs or drug names (returns the best match, if relevant)
        """
        if not isinstance(query, str):
            return None
        else:
            if query.startswith("DB"):
                return getattr(self.__db(), query)
            elif query in self.__drugNames:
                return getattr(self.__db(), self.__name2id[query])
            else:
                best_match = get_best_match(query, self.__drugNames)
                return getattr(self.__db(), self.__name2id[best_match])

    def get_drug(self, query):
        """
            Alias for get
        """
        return self.get(query)

    def get_current_release_url(self):
        if not self.update:
            try:
                import json

                with open("data/sources/sources.json", "r+") as infofile:
                    sources_data = json.load(infofile)
                    filename = list(sources_data["DrugBank"]["files"].keys())[0]
                    url = sources_data["DrugBank"]["files"][filename]["URL"]
                    version = sources_data["DrugBank"]["files"][filename]["version"]
                    self.v = re.findall("(.+)\    ", version)[
                        0
                    ]  # before four spaces (tab)
                    return url
            except Exception:
                log.warning("Unable to use local copy, forcing update")
                self._update = True
                return self.get_current_release_url()
        else:
            from bs4 import BeautifulSoup

            self.v = re.findall(
                "Version ([0-9]+\.[0-9]+\.[0-9]+) ",
                BeautifulSoup(
                    requests.get("https://go.drugbank.com/releases/latest").content,
                    "html5lib",
                ).head.title.text,
            )[0]
            url = f"https://go.drugbank.com/releases/{self.v.replace('.', '-')}/downloads/all-full-database"
            if os.path.isfile("data/sources/sources.json"):
                try:
                    import json

                    with open("data/sources/sources.json", "r+") as infofile:
                        sources_data = json.load(infofile)
                        local_url = sources_data["DrugBank"]["files"][
                            list(sources_data["DrugBank"]["files"].keys())[0]
                        ]["URL"]
                    if (
                        local_url == url
                    ):  # if there is not a newer version online it doesn't update the database
                        self._update = False
                except:
                    pass
            return url


# @profile()
class LINCS(Database):  # , metaclass=Singleton
    """
    LINCS reference class
    Contains info about gene expression profiles influenced by perturbing agents
    """

    def __init__(self, update=None, base_cell_lines=[], batch_size=None):
        Database.__init__(
            self,
            update=update,
            license="Freely Available",
            license_url="https://clue.io/connectopedia/publishing_with_geo_data, https://clue.io/connectopedia/data_redistribution, https://clue.io/connectopedia/clue_access_for_profits, https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
            requirements=[NCBI, DrugBank],
        )
        self.__base_cell_lines = set(base_cell_lines)
        self.__batch_size = batch_size
        if self.__batch_size:
            log.info(
                "'batch_size' provided. The database will be provided as a generator of pd.DataFrames and not as a single pd.DataFrame"
            )
        if self.__base_cell_lines == set():
            log.warning(
                "No base cell lines selected, all cell lines will be loaded and it may result in an OOM error!"
            )

        def get_hrefs_and_versions():
            import requests
            from bs4 import BeautifulSoup
            import re
            from dateutil.parser import parse as parsedate

            hrefs = {}
            versions = {}
            url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138"
            r = requests.get(url, allow_redirects=True)
            page = BeautifulSoup(r.content, "html5lib")
            supplementary_files = page.find(
                "table", attrs={"cellpadding": "2", "cellspacing": "2", "width": "600"}
            )
            hrefs["GSE70138"] = {
                f: h
                for h in [
                    td.a.get("href") for td in supplementary_files.findAll("td") if td.a
                ]
                for f in ["Level5", "cell_info", "gene_info", "pert_info", "sig_info"]
                if f"{f.replace('_', '%5F')}%5F" in h
            }
            versions["GSE70138"] = {
                k: re.findall("\%5F([0-9]{4}\%2D[0-9]{2}\%2D[0-9]{2})\%2E", href)[
                    0
                ].replace("%2D", "-")
                for k, href in hrefs["GSE70138"].items()
            }

            url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742"
            r = requests.get(url, allow_redirects=True)
            page = BeautifulSoup(r.content, "html5lib")
            supplementary_files = page.find(
                "table", attrs={"cellpadding": "2", "cellspacing": "2", "width": "600"}
            )
            hrefs["GSE92742"] = {
                f.replace("%2E", ""): h
                for h in [
                    td.a.get("href") for td in supplementary_files.findAll("td") if td.a
                ]
                for f in [
                    "Level5",
                    "cell_info",
                    "gene_info%2E",
                    "pert_info",
                    "sig_info",
                ]
                if f.replace("_", "%5F") in h
            }
            versions["GSE92742"] = parsedate(
                page.find("td", text="Last update date").find_next_sibling().text
            ).strftime("%Y-%m-%d")

            return hrefs, versions

        hrefs, versions = get_hrefs_and_versions()
        self.__GSE70138_cell_info = self._add_file(
            url=hrefs["GSE70138"]["cell_info"],
            retrieved_version=versions["GSE70138"]["cell_info"],
            sep="\t",
            update=update,
        )
        if self.__base_cell_lines:
            self.__cell_lines = set(
                self.__GSE70138_cell_info()[
                    self.__GSE70138_cell_info()["base_cell_id"].isin(
                        self.__base_cell_lines
                    )
                ].reset_index(drop=True)["cell_id"]
            )
        else:  # in case no cell lines are selected
            self.__cell_lines = set(self.__GSE70138_cell_info()["cell_id"])

        self.__GSE70138_gene_info = self._add_file(
            url=hrefs["GSE70138"]["gene_info"],
            retrieved_version=versions["GSE70138"]["gene_info"],
            sep="\t",
            update=update,
        )
        self.__BING_genes = set(
            self.__GSE70138_gene_info()[
                self.__GSE70138_gene_info()["pr_is_bing"] == "1"
            ]["pr_gene_id"]
        )
        self.__GSE70138_pert_info = self._add_file(
            url=hrefs["GSE70138"]["pert_info"],
            retrieved_version=versions["GSE70138"]["pert_info"],
            sep="\t",
            update=update,
        )
        self.__pertid2inchi = (
            self.__GSE70138_pert_info().set_index("pert_id")["inchi_key"].to_dict()
        )
        self.__GSE70138_sig_info = self._add_file(
            url=hrefs["GSE70138"]["sig_info"],
            retrieved_version=versions["GSE70138"]["sig_info"],
            sep="\t",
            update=update,
        )
        self.__sigid2pertid = (
            self.__GSE70138_sig_info().set_index("sig_id")["pert_id"].to_dict()
        )
        self.__sigid2cell = (
            self.__GSE70138_sig_info()[["sig_id", "cell_id"]]
            .set_index("sig_id")["cell_id"]
            .to_dict()
        )
        self.__sig_perturbed_by_trt = set(
            self.__GSE70138_sig_info()[
                self.__GSE70138_sig_info()["pert_type"].isin(("trt_cp", "trt_lig"))
            ]["sig_id"].values
        )  # keep only compounds, peptides and other biological agents used for treatment (https://clue.io/connectopedia/perturbagen_types_and_controls)
        self.__GSE92742_cell_info = self._add_file(
            url=hrefs["GSE92742"]["cell_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        if self.__base_cell_lines:
            self.__cell_lines = self.__cell_lines.union(
                set(
                    self.__GSE92742_cell_info()[
                        self.__GSE92742_cell_info()["base_cell_id"].isin(
                            self.__base_cell_lines
                        )
                    ].reset_index(drop=True)["cell_id"]
                )
            )
        else:  # in case no cell lines are selected
            self.__cell_lines = self.__cell_lines.union(
                set(self.__GSE92742_cell_info()["cell_id"])
            )

        self.__GSE92742_gene_info = self._add_file(
            url=hrefs["GSE92742"]["gene_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        self.__BING_genes = self.__BING_genes.union(
            set(
                self.__GSE92742_gene_info()[
                    self.__GSE92742_gene_info()["pr_is_bing"] == "1"
                ]["pr_gene_id"]
            )
        )
        self.__GSE92742_pert_info = self._add_file(
            url=hrefs["GSE92742"]["pert_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        self.__pertid2inchi.update(
            self.__GSE92742_pert_info().set_index("pert_id")["inchi_key"].to_dict()
        )
        self.__GSE92742_sig_info = self._add_file(
            url=hrefs["GSE92742"]["sig_info"],
            retrieved_version=versions["GSE92742"],
            sep="\t",
            update=update,
        )
        self.__sigid2pertid.update(
            self.__GSE92742_sig_info().set_index("sig_id")["pert_id"].to_dict()
        )
        self.__sigid2cell.update(
            (
                self.__GSE92742_sig_info()[["sig_id", "cell_id"]]
                .set_index("sig_id")["cell_id"]
                .to_dict()
            )
        )
        self.__sig_perturbed_by_trt = self.__sig_perturbed_by_trt.union(
            set(
                self.__GSE92742_sig_info()[
                    self.__GSE92742_sig_info()["pert_type"].isin(("trt_cp", "trt_lig"))
                ]["sig_id"].values
            )
        )  # keep only compounds, peptides and other biological agents used for treatment (https://clue.io/connectopedia/perturbagen_types_and_controls)
        self.__sigid2inchi = {
            k: self.__pertid2inchi[v]
            for k, v in self.__sigid2pertid.items()
            if v in self.__pertid2inchi
        }
        self.__sigid2DBid = {
            k: self.__DrugBank.inchiKeyBase2id[v.split("-")[0]]
            for k, v in self.__sigid2inchi.items()
            if v.split("-")[0] in self.__DrugBank.inchiKeyBase2id
        }

        self.__sigid2DBname = {
            k: self.__DrugBank.inchiKeyBase2name[v.split("-")[0]]
            for k, v in self.__sigid2inchi.items()
            if v.split("-")[0] in self.__DrugBank.inchiKeyBase2name
        }

        def custom_read_expression_profiles(filepath):
            filepath = filepath.rstrip(".gz")
            if not os.path.isfile(filepath):
                import gzip
                import shutil

                with gzip.open(filepath + ".gz", "rb") as infile:
                    with open(filepath, "wb") as outfile:
                        shutil.copyfileobj(infile, outfile)
            from cmapPy.pandasGEXpress.parse import parse as parse_gctx

            cols = parse_gctx(filepath, col_meta_only=True)
            cols["col_id"] = range(len(cols))
            cols["col_name"] = cols.index
            cols.set_index("col_id", inplace=True)
            cols = cols[
                cols["col_name"].map(self.__sigid2cell.get).isin(self.__cell_lines)
            ]  # filter by selected cell lines
            cols = cols[
                cols["col_name"].isin(self.__sig_perturbed_by_trt)
            ]  # keep only signatures perturbed by compounds, peptides or other biological agents used for treatment
            cols = cols[
                cols["col_name"].isin(self.__sigid2DBid)
            ]  # keep only signatures perturbed by entities available in drugbank

            if self.__batch_size != None:

                batches = (
                    np.array_split(cols.index, len(cols) // self.__batch_size)
                    if len(cols) > self.__batch_size
                    else [cols.index]
                )

                return (
                    parse_gctx(filepath, cidx=batch,).data_df.loc[
                        list(self.__BING_genes)
                    ]
                    for batch in batches
                )
            else:
                return parse_gctx(filepath,).data_df.loc[list(self.__BING_genes)]

        self.__GSE70138_level5 = self._add_file(
            url=hrefs["GSE70138"]["Level5"],
            retrieved_version=versions["GSE70138"]["Level5"],
            custom_read_function=custom_read_expression_profiles,
            update=update,
        )
        self.__GSE92742_level5 = self._add_file(
            url=hrefs["GSE92742"]["Level5"],
            retrieved_version=versions["GSE92742"],
            custom_read_function=custom_read_expression_profiles,
            update=update,
        )

    @property
    def base_cell_lines(self):
        return self.__base_cell_lines.copy()

    @property
    def cell_lines(self):
        return self.__cell_lines.copy()

    @property
    def BING_genes(self):
        return self.__BING_genes.copy()

    @property
    def sig_perturbed_by_trt(self):
        """
            Names of signatures perturbed by compounds, peptides or other biological agents used for treatment
        """
        return self.__sig_perturbed_by_trt.copy()

    @property
    def sigid2cell(self):
        return self.__sigid2cell.copy()

    def get_cell_by_sigid(self, sigid):
        return self.__sigid2cell.get(sigid)

    @property
    def sigid2pertid(self):
        return self.__sigid2pertid.copy()

    def get_pertid_by_sigid(self, sigid):
        return self.__sigid2pertid.get(sigid)

    @property
    def sigid2DBid(self):
        return self.__sigid2DBid.copy()

    def get_DBid_by_sigid(self, sigid):
        return self.__sigid2DBid.get(sigid)

    @property
    def sigid2DBname(self):
        return self.__sigid2DBname.copy()

    def get_DBname_by_sigid(self, sigid):
        return self.__sigid2DBname.get(sigid)

    @property
    def GSE70138_cell_info(self):
        return self.__GSE70138_cell_info().copy()

    @property
    def GSE92742_cell_info(self):
        return self.__GSE92742_cell_info().copy()

    @property
    def GSE70138_gene_info(self):
        return self.__GSE70138_gene_info().copy()

    @property
    def GSE92742_gene_info(self):
        return self.__GSE92742_gene_info().copy()

    @property
    def GSE70138_pert_info(self):
        return self.__GSE70138_pert_info().copy()

    @property
    def GSE92742_pert_info(self):
        return self.__GSE92742_pert_info().copy()

    @property
    def GSE70138_sig_info(self):
        return self.__GSE70138_sig_info().copy()

    @property
    def GSE92742_sig_info(self):
        return self.__GSE92742_sig_info().copy()

    @property
    def GSE70138_level5(self):
        if self.__batch_size:
            content = self.__GSE70138_level5()
            self.__GSE70138_level5.read()
            return content
        else:
            return self.__GSE70138_level5().copy()

    @property
    def GSE92742_level5(self):
        if self.__batch_size:
            content = self.__GSE92742_level5()
            self.__GSE92742_level5.read()
            return content
        else:
            return self.__GSE92742_level5().copy()

    @property
    def database(self):
        if self.__batch_size:
            from itertools import chain

            return chain(self.GSE70138_level5, self.GSE92742_level5)
        else:
            return self.GSE70138_level5.join(self.GSE92742_level5)


# @profile()
class HPO(Database, metaclass=Singleton):
    """
    Human Phenotype Ontology reference class
    Contains info about phenotypes
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license_url="https://hpo.jax.org/app/license",
            requirements=[NCBI],
        )
        self.__db = self._add_file(url="http://purl.obolibrary.org/obo/hp.obo")
        self.__phenotype2name = set()
        for frame in tqdm(
            self.__db(), desc=f"Collecting info from {self.__db.filename}"
        ):
            for clause in frame:
                if isinstance(clause, fastobo.term.NameClause):
                    self.__phenotype2name.add((f"HP:{frame.id.local}", clause.name))

        def read_gene2phenotype(filepath):
            with open(filepath, "r") as infile:
                cols = infile.readline().strip().lstrip("#Format: ").split("<tab>")
            return pd.read_csv(filepath, sep="\t", skiprows=1, names=cols,)

        self.__gene2phenotype = self._add_file(
            url="http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
            custom_read_function=read_gene2phenotype,
        )

        self.__geneSymbol2hpo = {
            symbol: set(group["HPO-Term-ID"].values)
            for symbol, group in self.__gene2phenotype()[
                ["entrez-gene-symbol", "HPO-Term-ID"]
            ].groupby("entrez-gene-symbol")
        }

        self.__geneId2hpo = {
            id: set(group["HPO-Term-ID"].values)
            for id, group in self.__gene2phenotype()[
                ["entrez-gene-id", "HPO-Term-ID"]
            ].groupby("entrez-gene-id")
        }

        # dicts for info retrieval
        self.__phenotype2name_asdict = dict(self.__phenotype2name)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def ontology(self):
        return self.__db().copy()

    @property
    def geneId2hpo(self):
        return self.__geneId2hpo.copy()

    @property
    def geneSymbol2hpo(self):
        return self.__geneSymbol2hpo.copy()

    def get_name(self, id):
        """Returns the name of a HPO id"""
        if not id.startswith("HPO:"):
            id = f"HPO:{id}"
        return self.__phenotype2name_asdict.get(id, f"Id {id} not found")


# @profile()
class OMIM(Database, metaclass=Singleton):
    """
    OMIM reference class
    Contains info about Human Genes and Genetic Disorders
    """

    def __init__(self, update=None, only_query=False):

        try:
            self.__apiKey = os.environ["OMIM_APIKEY"]
        except KeyError:
            try:
                from dotenv import dotenv_values

                credentials = dotenv_values()
                self.__apiKey = credentials["OMIM_APIKEY"]
            except:
                raise RuntimeError(
                    "No OMIM API KEY found, "
                    "Human Genes and Genetic Disorders information will not be collected"
                )

        Database.__init__(
            self,
            update=update,
            license="Restricted Availability",
            license_url="https://www.omim.org/help/copyright",
            registration_required=True,
            requirements=[NCBI],
        )

        if not only_query:
            self._get_files()

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def mim2gene(self):
        return self.__mim2gene().copy()

    @property
    def mimTitles(self):
        return self.__mimTitles().copy()

    @property
    def genemap2(self):
        return self.__genemap2().copy()

    @property
    def morbidmap(self):
        return self.__morbidmap().copy()

    def search(self, query, geneMap=False):
        import urllib

        query = urllib.parse.quote(query)
        limit = 20 if geneMap else 100
        url = (
            lambda: f"https://api.omim.org/api/entry/search?search=%22{query}%22&start={start}&limit={limit}&include=geneMap&format=json&apiKey={self.__apiKey}"
            if geneMap
            else f"https://api.omim.org/api/entry/search?search=%22{query}%22&start={start}&limit={limit}&format=json&apiKey={self.__apiKey}"
        )  # it's a lambda function because in this way it can be updated

        start = 0
        search_response = requests.get(url()).json()["omim"]["searchResponse"]
        for start in tqdm(
            range(limit, limit * (search_response["totalResults"] // limit) + 1, limit)
        ):
            tmp_json = requests.get(url()).json()["omim"]["searchResponse"]
            search_response["endIndex"] = tmp_json["endIndex"]
            search_response["searchTime"] += tmp_json["searchTime"]
            search_response["entryList"] += tmp_json["entryList"]

        return search_response

    def get_disease_genes(self, disease):
        genes = []
        for entry in self.search(disease, geneMap=True)["entryList"]:
            entry = entry["entry"]
            if "prefix" in entry:
                if entry["prefix"] in "*+%" and "geneMap" in entry:
                    genes.append(
                        self.__NCBI.check_symbol(
                            entry["geneMap"].get("approvedGeneSymbols"),
                            aliases=entry["geneMap"].get("geneSymbols").split(", "),
                        )
                    )
                elif entry["prefix"] == "#" and "phenotypeMapList" in entry:
                    for map in entry["phenotypeMapList"]:
                        genes.append(
                            self.__NCBI.check_symbol(
                                map["phenotypeMap"].get("approvedGeneSymbols"),
                                aliases=map["phenotypeMap"]
                                .get("geneSymbols")
                                .split(", "),
                            )
                        )
        genes = list({gene for gene in genes if gene})
        return genes

    def _get_files(self):
        self.__mim2gene = self._add_file(
            url="https://omim.org/static/omim/data/mim2gene.txt",
            names=[
                "MIM Number",
                "MIM Entry Type",
                "Entrez Gene ID",
                "Approved Gene Symbol",
                "Ensembl Gene ID",
            ],
            comment="#",
        )

        self.__mimTitles = self._add_file(
            url=f"https://data.omim.org/downloads/{self.__apiKey}/mimTitles.txt",
            censored_url="https://data.omim.org/downloads/{apiKey}/mimTitles.txt",
            names=[
                "Prefix",
                "MIM Number",
                "Preferred Title",
                "Alternative Title(s)",
                "Included Title(s)",
            ],
            comment="#",
        )

        self.__genemap2 = self._add_file(
            url=f"https://data.omim.org/downloads/{self.__apiKey}/genemap2.txt",
            censored_url="https://data.omim.org/downloads/{apiKey}/genemap2.txt",
            names=[
                "Chromosome",
                "Genomic Position Start",
                "Genomic Position End",
                "Cyto Location",
                "Computed Cyto Location",
                "MIM Number",
                "Gene Symbols",
                "Gene Name",
                "Approved Gene Symbol",
                "Entrez Gene ID",
                "Ensembl Gene ID",
                "Comments",
                "Phenotypes",
                "Mouse Gene Symbol/ID",
            ],
            comment="#",
            index_col=False,
        )

        self.__morbidmap = self._add_file(
            url=f"https://data.omim.org/downloads/{self.__apiKey}/morbidmap.txt",
            censored_url="https://data.omim.org/downloads/{apiKey}/morbidmap.txt",
            names=["Phenotype", "Gene Symbols", "MIM Number", "Cyto Location"],
            comment="#",
        )

        self.__morbidmap = self._add_file(
            url=f"https://data.omim.org/downloads/{self.__apiKey}/morbidmap.txt",
            censored_url="https://data.omim.org/downloads/{apiKey}/morbidmap.txt",
            names=["Phenotype", "Gene Symbols", "MIM Number", "Cyto Location"],
            comment="#",
        )


################################################################################
#
#                                 Interactome
#
################################################################################

# @profile()
class APID(Database, metaclass=Singleton):
    """
        APID reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC-BY-NC",
            license_url="http://cicblade.dep.usal.es:8080/APID/init.action#subtab4",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="http://cicblade.dep.usal.es:8080/APID/InteractionsTABplain.action",
            post=True,
            data_to_send={
                "interactomeTaxon": "9606",
                "interactomeTaxon1": "9606",
                "quality": 1,
                "interspecies": "NO",
            },
            final_columns=["GeneName_A", "GeneName_B"],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions["subject"] = self.__interactions["GeneName_A"].apply(
            self.__NCBI.check_symbol
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions["GeneName_B"].apply(
            self.__NCBI.check_symbol
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class BioGRID(Database, metaclass=Singleton):
    """
        BioGRID reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="MIT",
            license_url="https://downloads.thebiogrid.org/BioGRID",
            requirements=[NCBI],
        )

        def read_biogrid_file(filepath):
            import zipfile

            with zipfile.ZipFile(filepath) as z:
                with z.open(f"BIOGRID-ORGANISM-Homo_sapiens-{self.v}.tab.txt") as f:
                    return pd.read_csv(f, skiprows=35, sep="\t", dtype=str).query(
                        "ORGANISM_A_ID == '9606' and ORGANISM_B_ID == '9606'"
                    )

        self.__db = self._add_file(
            url=self.get_current_release_url(),
            retrieved_version=self.v,  # retrieved by get_current_release_url
            custom_read_function=read_biogrid_file,
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()[
            [
                "OFFICIAL_SYMBOL_A",
                "OFFICIAL_SYMBOL_B",
                "ALIASES_FOR_A",
                "ALIASES_FOR_B",
                "source",
            ]
        ]
        self.__interactions["subject"] = self.__interactions[
            ["OFFICIAL_SYMBOL_A", "ALIASES_FOR_A"]
        ].apply(
            lambda x: self.__NCBI.check_symbol(x[0], x[1].split("|")), axis=1
        )  # checks symbol A symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions[
            ["OFFICIAL_SYMBOL_B", "ALIASES_FOR_B"]
        ].apply(
            lambda x: self.__NCBI.check_symbol(x[0], x[1].split("|")), axis=1
        )  # hecks symbol B symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()

    def get_current_release_url(self):
        if not self.update:
            try:
                import json

                with open("data/sources/sources.json", "r+") as infofile:
                    sources_data = json.load(infofile)
                    filename = list(sources_data["BioGRID"]["files"].keys())[0]
                    url = sources_data["BioGRID"]["files"][filename]["URL"]
                    version = sources_data["BioGRID"]["files"][filename]["version"]
                    self.v = re.findall("(.+)\    ", version)[
                        0
                    ]  # before four spaces (tab)
                    return url
            except Exception:
                log.warning("Unable to use local copy, forcing update")
                self._update = True
                return self.get_current_release_url()
        else:
            from bs4 import BeautifulSoup

            r = requests.get("https://downloads.thebiogrid.org/BioGRID/")
            page = r.content
            soup = BeautifulSoup(page, "html5lib")
            href = soup.find("a", string="Current-Release")["href"]
            self.v = re.findall("BIOGRID-(.+)/", href)[0]
            url = f"https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-{self.v}/BIOGRID-ORGANISM-{self.v}.tab.zip"
            return url


# @profile()
class HuRI(Database, metaclass=Singleton):
    """
        HuRI reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="http://www.interactome-atlas.org/download",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="http://www.interactome-atlas.org/data/HuRI.tsv",
            names=["protein1", "protein2"],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions = self.__interactions.merge(
            self.__NCBI.ensembl2ncbi.rename(
                columns={"Ensembl": "protein1", "NCBI": "subject"}
            ),
            on="protein1",
            how="left",
        )  # converts Ensembl ids to NCBI
        self.__interactions["subject"] = self.__interactions["subject"].apply(
            self.__NCBI.check_symbol
        )  # checks protein1 symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions = self.__interactions.merge(
            self.__NCBI.ensembl2ncbi.rename(
                columns={"Ensembl": "protein2", "NCBI": "object"}
            ),
            on="protein2",
            how="left",
        )  # converts Ensembl ids to NCBI
        self.__interactions["object"] = self.__interactions["object"].apply(
            self.__NCBI.check_symbol
        )  # checks protein2 symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class InnateDB(Database, metaclass=Singleton):
    """
        InnateDB reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="DESIGN SCIENCE LICENSE",
            license_url="https://www.innatedb.com/license.jsp",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="https://www.innatedb.com/download/interactions/all.mitab.gz",
            usecols=[
                "alias_A",
                "alias_B",
                "ncbi_taxid_A",
                "ncbi_taxid_B",
                "confidence_score",
            ],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db().query(
            "ncbi_taxid_A == 'taxid:9606(Human)' and ncbi_taxid_B == 'taxid:9606(Human)'"
        )  # check if actually from Homo sapiens
        self.__interactions = self.__interactions[
            self.__interactions["confidence_score"].apply(
                lambda s: int(re.findall("np:(.+)\|", s)[0]) >= 1
            )
        ]  # checks that there is at least one publication supporting the interaction that has never been used to support any other interaction #http://wodaklab.org/iRefWeb/faq
        self.__interactions["subject"] = self.__interactions["alias_A"].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall("hgnc:(.+)\(display_short\)|$", s)[0]
            )
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions["alias_B"].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall("hgnc:(.+)\(display_short\)|$", s)[0]
            )
        )  # checks the symbol of the second gene
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = self.__interactions[["subject", "object", "source"]]
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            ),
            on="subject",
            how="left",
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            ),
            on="object",
            how="left",
        )  # gets object name
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class INstruct(Database, metaclass=Singleton):
    """
        INstruct reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="All rights reserved (Authorization obtained by e-mail contact with Haiyuan Yu <haiyuan.yu@cornell.edu>)",
            license_url="http://instruct.yulab.org/about.html",
            requirements=[NCBI],
        )
        self.__db = self._add_file(
            url="http://instruct.yulab.org/download/sapiens.sin",
            usecols=["ProtA[Official Symbol]", "ProtB[Official Symbol]"],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions["subject"] = self.__interactions[
            "ProtA[Official Symbol]"
        ].apply(
            self.__NCBI.check_symbol
        )  # checks protA symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions[
            "ProtB[Official Symbol]"
        ].apply(
            self.__NCBI.check_symbol
        )  # checks protB symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class IntAct(Database, metaclass=Singleton):
    """
        IntAct reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC-BY 4.0",
            license_url="https://www.ebi.ac.uk/intact/resources/overview",
            requirements=[NCBI],
        )

        def read_intact_file(filepath):
            import zipfile

            with zipfile.ZipFile(filepath) as z:
                with z.open("intact.txt") as f:
                    return pd.read_csv(f, sep="\t", dtype=str)

        self.__db = self._add_file(
            url="http://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip",
            custom_read_function=read_intact_file,
            usecols=[
                "Alias(es) interactor A",
                "Alias(es) interactor B",
                "Taxid interactor A",
                "Taxid interactor B",
                "Confidence value(s)",
            ],
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db()
        self.__interactions = self.__interactions[
            (
                self.__interactions["Taxid interactor A"].str.contains(
                    "taxid:9606\(Homo sapiens\)"
                )
            )
            & (
                self.__interactions["Taxid interactor B"].str.contains(
                    "taxid:9606\(Homo sapiens\)"
                )
            )
        ]  # check if actually from Homo sapiens (both interactor A and B)
        self.__interactions = self.__interactions[
            self.__interactions["Confidence value(s)"].apply(
                lambda s: float(re.findall("intact-miscore:(.+)", s)[0]) >= 0.6
            )
        ]  # threshold for high confidence https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4316181/pdf/bau131.pdf
        self.__interactions["subject"] = self.__interactions[
            "Alias(es) interactor A"
        ].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall(":([a-zA-Z]+)\(gene name\)|$", s)[0],
                [
                    re.findall(":(.+)\(|$", alias)[0]
                    for alias in s.split("|")
                    if "(gene name synonym)" in alias
                ],
            )
        )  # checks the symbol of the first gene
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions[
            "Alias(es) interactor B"
        ].apply(
            lambda s: self.__NCBI.check_symbol(
                re.findall(":([a-zA-Z]+)\(gene name\)|$", s)[0],
                [
                    re.findall(":(.+)\(|$", alias)[0]
                    for alias in s.split("|")
                    if "(gene name synonym)" in alias
                ],
            )
        )  # checks the symbol of the second gene
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = self.__interactions[
            ["subject", "object", "source"]
        ].drop_duplicates(ignore_index=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            ),
            on="subject",
            how="left",
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            ),
            on="object",
            how="left",
        )  # gets object name
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class SignaLink(Database, metaclass=Singleton):
    """
        SignaLink reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY-NC-SA 3.0",
            license_url="http://signalink.org/faq",
            requirements=[NCBI],
        )

        def read_signalink_file(filepath):
            import tarfile
            import json

            with tarfile.open(filepath) as archive:
                edges_member = [file for file in archive.getnames() if "edges" in file][
                    0
                ]
                nodes_member = [file for file in archive.getnames() if "nodes" in file][
                    0
                ]
                raw_edges = json.load(archive.extractfile(edges_member))
                raw_nodes = json.load(archive.extractfile(nodes_member))
            taxon = {
                node["displayedName"]: int(node["taxon"]["id"]) for node in raw_nodes
            }
            accepted_dbs = (
                "SignaLink",
                "ACSN",
                "InnateDB",
                "Signor",
                "PhosphoSite",
                "TheBiogrid",
                "ComPPI",
                "HPRD",
                "IntAct",
                "OmniPath",
            )  # PSP == Potential Scaffold Proteins
            edges = set()
            for edge in raw_edges:
                if len(
                    {
                        db["value"]
                        for db in edge["sourceDatabases"]
                        if db["value"] in accepted_dbs
                    }
                ):  # only trusted databases for proteins
                    source = edge["sourceDisplayedName"]
                    target = edge["targetDisplayedName"]
                    if (
                        taxon[source] == 9606 and taxon[target] == 9606
                    ):  # filters for homo sapiens
                        source = self.__NCBI.check_symbol(
                            source
                        )  # checks the symbol of the source
                        target = self.__NCBI.check_symbol(
                            target
                        )  # checks the symbol of the target
                        if source and target:
                            edges.add(
                                (
                                    source,
                                    target,
                                    edge["sourceFullName"],
                                    edge["targetFullName"],
                                    ", ".join(
                                        {db["value"] for db in edge["sourceDatabases"]}
                                    ),
                                )
                            )
            return (
                pd.DataFrame(
                    edges,
                    columns=[
                        "sourceSymbol",
                        "targetSymbol",
                        "sourceName",
                        "targetName",
                        "database",
                    ],
                )
                .drop_duplicates(ignore_index=True)
                .astype(
                    {
                        "sourceSymbol": "string",
                        "targetSymbol": "string",
                        "sourceName": "string",
                        "targetName": "string",
                        "database": "category",
                    }
                )
            )

        self.__db = self._add_file(  # http:://signalink.org/download/521c18e9ea050e801018
            url="http://signalink.org/slk3db_dump_json.tgz",  # http://signalink.org/download/33a38c88031e461c8c29 # http://signalink.org/download/e64ffdd983087ea7c794 csv # http:://signalink.org/download/33a38c88031e461c8c29 psimitab
            custom_read_function=read_signalink_file,
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db().rename(
            columns={"sourceSymbol": "subject", "targetSymbol": "object"}
        )[["subject", "object", "source"]]
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            ),
            on="subject",
            how="left",
        )  # gets subject name (names not provided by NCBI or HGNC are not trusted)
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            ),
            on="object",
            how="left",
        )  # gets object name (names not provided by NCBI or HGNC are not trusted)
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()


# @profile()
class STRING(Database, metaclass=Singleton):
    """
        STRING reference class
        Contains info about protein-protein interactions
    """

    def __init__(self, update=None):
        Database.__init__(
            self,
            update=update,
            license="CC BY 4.0",
            license_url="https://string-db.org/cgi/access?footer_active_subpage=licensing",
            requirements=[NCBI],
        )
        self.__symbol2string = self._add_file(
            url="https://string-db.org/mapping_files/STRING_display_names/human.name_2_string.tsv.gz",
            skiprows=1,
            names=["NCBI taxid", "geneSymbol", "STRING"],
            final_columns=["geneSymbol", "STRING"],
        )
        self.__symbol2string_asdict = (
            self.__symbol2string().set_index("geneSymbol")["STRING"].to_dict()
        )
        self.__string2symbol_asdict = (
            self.__symbol2string().set_index("STRING")["geneSymbol"].to_dict()
        )
        self.__db = self._add_file(
            url=self.get_current_release_url(),
            sep=" ",
            retrieved_version=self.v,  # retrieved by get_current_release_url
            dtype={"protein1": str, "protein2": str, "combined_score": int},
        )
        log.info(f"Retrieving interactions from {self.__class__.__name__}")
        self.__interactions = self.__db().query(
            "combined_score >= 700"
        )  # threshold for high confidence https://string-db.org/help/faq/#how-to-extract-high-confidence-07-interactions-from-information-on-combined-score-in-proteinlinkstxtgz
        self.__interactions["subject"] = self.__interactions["protein1"].apply(
            lambda s: self.__NCBI.check_symbol(self.get_symbol_by_string(s))
        )  # checks protA symbol
        self.__interactions.dropna(subset=["subject"], inplace=True)
        self.__interactions["object"] = self.__interactions["protein2"].apply(
            lambda s: self.__NCBI.check_symbol(self.get_symbol_by_string(s))
        )  # checks protB symbol
        self.__interactions.dropna(subset=["object"], inplace=True)
        self.__interactions = pd.concat(
            [
                self.__interactions,
                self.__interactions.rename(
                    columns={"subject": "object", "object": "subject"}
                ),
            ]
        ).drop_duplicates(
            ignore_index=True
        )  # considers also inverse interactions
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "subject", "geneName": "subjectName"}
            )
        )  # gets subject name
        self.__interactions = self.__interactions.merge(
            self.__NCBI.gene2name.rename(
                columns={"geneSymbol": "object", "geneName": "objectName"}
            )
        )  # gets object name
        self.__interactions["relation"] = ["interacts_with"] * len(self.__interactions)
        self.__interactions["subjectType"] = ["protein"] * len(self.__interactions)
        self.__interactions["objectType"] = ["protein"] * len(self.__interactions)
        self.__interactions = (
            self.__interactions[
                [
                    "subject",
                    "relation",
                    "object",
                    "subjectName",
                    "objectName",
                    "subjectType",
                    "objectType",
                    "source",
                ]
            ]
            .drop_duplicates(ignore_index=True)
            .astype(
                {
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                }
            )
        )

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def database(self):
        return self.__db().copy()

    @property
    def interactions(self):
        return self.__interactions.copy()

    @property
    def symbol2string(self):
        return self.__symbol2string().copy()

    @property
    def symbol2string_asdict(self):
        return self.__symbol2string_asdict.copy()

    @property
    def string2symbol_asdict(self):
        return self.__string2symbol_asdict.copy()

    def get_string_by_symbol(self, symbol):
        return self.__symbol2string_asdict.get(symbol, f"{symbol} Not Found")

    def get_symbol_by_string(self, string):
        if not string.startswith("9606."):
            string = "9606." + string
        return self.__string2symbol_asdict.get(string, f"{string} Not Found")

    def get_current_release_url(self):
        if not self.update:
            try:
                import json

                with open("data/sources/sources.json", "r+") as infofile:
                    sources_data = json.load(infofile)
                    filename = [
                        f
                        for f in sources_data["STRING"]["files"]
                        if "protein.links" in f
                    ][0]
                    url = sources_data["STRING"]["files"][filename]["URL"]
                    self.v = re.findall("v(.+)\.txt", filename)[0]
                    return url
            except Exception:
                log.warning("Unable to use local copy, forcing update")
                self._update = True
                return self.get_current_release_url()
        else:
            import requests
            from bs4 import BeautifulSoup

            r = requests.get("https://stringdb-static.org/download/")
            page = r.content
            soup = BeautifulSoup(page, "html5lib")
            self.v = [a for a in soup.findAll("a") if "protein.links.v" in a["href"]][
                0
            ].text[
                15:-1
            ]  # current release version
            url = f"https://stringdb-static.org/download/protein.links.v{self.v}/9606.protein.links.v{self.v}.txt.gz"
            return url
