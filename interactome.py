import pandas as pd

from utils import integrate_dataframes, profile, Singleton
import databases as dbs

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "interactome"
log = logging.getLogger(logging_name)


################################################################################
#
#                   Human Protein-Protein Interaction Network
#
################################################################################


# @profile()
class PPI(metaclass=Singleton):
    """
    Collect and merge protein-protein interactions for Homo sapiens
    Sources are APID, BioGRID, HuRI, InnateDB, INstruct, IntAct, SignLink, STRING
    """

    def __init__(self, update=None):
        log = logging.getLogger("interactome:PPI")
        log.info("Collecting protein-protein interactions")
        self.__update = update
        log.info(
            f"Retrieving source databases of protein-protein interactions for {self.__class__.__name__}"
        )
        self.__sources = {
            db.name: db
            for db in [
                dbs.APID(update=self.__update),
                dbs.BioGRID(update=self.__update),
                dbs.HuRI(update=self.__update),
                dbs.InnateDB(update=self.__update),
                dbs.INstruct(update=self.__update),
                dbs.IntAct(update=self.__update),
                dbs.SignaLink(update=self.__update),
                dbs.STRING(update=self.__update),
            ]
        }
        self.__versions = {db.name: db.version for db in self.__sources.values()}

        # save files only if there are updates in the databases
        if any([db.updated for db in self.__sources.values()]):
            log.info(f"Integrating source databases for {self.__class__.__name__}")
            self.__interactome = integrate_dataframes(
                [db.interactions for db in self.__sources.values()],
                columns_to_join=["source"],
            )
            self.__interactome.to_csv(
                "data/sources/interactome.tsv.gz", sep="\t", compression="gzip",
            )
            self.__interactome[["subject", "object"]].rename(
                columns={"subject": "source", "object": "target"}
            ).to_csv(
                "data/sources/interactome_slim.tsv.gz",
                sep="\t",
                compression="gzip",
                index=False,
            )
        else:
            self.__interactome = pd.read_csv(
                "data/sources/interactome.tsv.gz",
                sep="\t",
                compression="gzip",
                index_col=0,
                dtype={
                    "subject": "string",
                    "relation": "category",
                    "object": "string",
                    "subjectName": "string",
                    "objectName": "string",
                    "subjectType": "category",
                    "objectType": "category",
                    "source": "string",
                },
            )

        self.__proteins = pd.concat(
            [
                self.__interactome[["subject", "subjectName"]].rename(
                    columns={"subject": "proteinSymbol", "subjectName": "proteinName"}
                ),
                self.__interactome[["object", "objectName"]].rename(
                    columns={"object": "proteinSymbol", "objectName": "proteinName"}
                ),
            ]
        ).drop_duplicates(ignore_index=True)

        log.info(f"{self.__class__.__name__} ready!")

    @property
    def interactome(self):
        return self.__interactome.copy()

    @property
    def proteins(self):
        return self.__proteins.copy()

    @property
    def sources(self):
        return self.__sources.copy()

    @property
    def version(self):
        print(f"{self.__class__.__name__} sources versions:")
        for key, value in self.__versions.items():
            print(f"\t{key}:\t{value}")

    @property
    def versions(self):
        return self.__versions

    def __call__(self):
        return self.__interactome.copy()

    def __repr__(self):
        return "Human Protein-Protein Interactome"

    def __str__(self):
        return f"Human Protein-Protein Interactome\nTotal Interactions: {len(self.__interactome)}\nTotal Proteins: {len(self.__proteins)}"


if __name__ == "__main__":
    ppi = PPI()
