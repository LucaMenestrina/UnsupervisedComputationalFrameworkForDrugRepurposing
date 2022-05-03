import pandas as pd

import requests
from bs4 import BeautifulSoup
import ftplib
import urllib
import re
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse as parsedate
import os
import gzip
import json
import fastobo
import inflection

import ray

import cProfile
import pstats
from functools import wraps

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "utils"
log = logging.getLogger(logging_name)


################################################################################
#
#                                   Utilies
#
################################################################################


def profile(sort_by="tottime", lines=20, strip_dirs=True):
    """A time profiler decorator
    adapted from https://towardsdatascience.com/how-to-profile-your-code-in-python-e70c834fad89 and http://code.activestate.com/recipes/577817-profile-decorator/
    """

    def accessory(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profile = cProfile.Profile()
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            stats = pstats.Stats(profile)
            if strip_dirs:
                stats = stats.strip_dirs()
            stats.sort_stats(sort_by)
            print(f"{func.__name__} Profile")
            stats.print_stats(lines)
            return result

        return wrapper

    return accessory


def download(
    url,
    folder,
    update=None,
    version_marker=None,
    retrieved_version=None,
    session=None,
    post=False,
    data_to_send=None,
):
    log = logging.getLogger("utils:download")
    log.debug("Requesting header for %s" % folder)
    if not session:
        session = requests.Session()
    if url.startswith("ftp:"):  # for ftp servers
        if "//" in url:
            url = url.split("//")[1]
        splitted = url.split("/")
        server = splitted[0]
        directory = "/".join(splitted[1:-1])
        filename = urllib.parse.unquote(splitted[-1])
        ftp = ftplib.FTP(server)  # , timeout=30
        ftp.login()
        ftp.cwd(directory)
        ref_date = parsedate(
            ftp.voidcmd(f"MDTM {filename}").split(" ")[1].strip()
        ).astimezone(timezone.utc)
        filepath = os.path.join("data/sources", folder, filename)
    elif url.startswith("http") and not post:
        h = session.head(url)  # , timeout=30
        h.raise_for_status()
        cd = h.headers.get("Content-Disposition")
        if cd:
            filename = re.findall("filename=(.+)", cd)[0]
        else:
            filename = url.split("/")[-1]
        filename = urllib.parse.unquote(filename.strip("\"'"))
        filepath = os.path.join("data/sources", folder, filename)
        try:
            ref_date = parsedate(h.headers.get("last-modified")).astimezone(
                timezone.utc
            )
        except:
            ref_date = None
            if update == None:
                update = True
    elif url.startswith("http") and post:
        log.info("Downloading data for %s" % folder)
        r = requests.post(url, allow_redirects=True, data=data_to_send)  # , timeout=30
        r.raise_for_status()
        cd = r.headers.get("Content-Disposition")
        if cd:
            filename = re.findall("filename=(.+)", cd)[0]
        else:
            filename = url.split("/")[-1]
        filename = urllib.parse.unquote(filename.strip("\"'"))
        filepath = os.path.join("data/sources", folder, filename)
        try:
            ref_date = parsedate(r.headers.get("last-modified")).astimezone(
                timezone.utc
            )
        except:
            ref_date = None
            if update == None:
                update = True
    if os.path.isfile(filepath):
        file_date = datetime.fromtimestamp(os.path.getmtime(filepath)).astimezone(
            timezone.utc
        )
    else:
        update = True
    if update or (update == None and ref_date > file_date):
        log.info(f"Downloading {filename} for {folder}")
        if not os.path.isdir(os.path.join("data/sources", folder)):
            os.mkdir(os.path.join("data/sources", folder))
        if "http" in url:
            if not post:
                r = session.get(url, allow_redirects=True)
                r.raise_for_status()
            with open(filepath, "wb") as output:
                output.write(r.content)
        elif "ftp" in url:
            with open(filepath, "wb") as file:
                ftp.retrbinary(
                    f"RETR {filename}",
                    file.write,
                    blocksize=32
                    * 1024
                    * 1024,  # blocksize suggested by NCBI for their database (https://ftp.ncbi.nlm.nih.gov/README.ftp)
                )
            ftp.quit()
        update = True  # to track if updated
    version = get_version(
        filepath=filepath,
        marker=version_marker,
        ref_date=ref_date,
        retrieved_version=retrieved_version,
    )
    return filepath, filename, version, update


def camelize(string):
    # for type style name consistency
    # addedd dash and space compatibility to inflection.camelize function

    if " " in string:
        string = string.replace(" ", "_")
    if "-" in string:
        string = string.replace("-", "_")

    return inflection.camelize(string, uppercase_first_letter=False)
    # # deprecated --> substituted by inflection package
    # return (
    #     string[0].lower()
    #     + "".join([word.capitalize() for word in re.split(" |_|-",string)])[1:]
    # )


# @profile()
def integrate_dataframes(
    dataframes, common_columns=None, columns_to_join=["source"], separator="|"
):
    """
        Integrates the provided DataFrames

        Returns a DataFrame merged on 'common_columns'
        and joins the content of 'columns_to_join'
    """
    if not isinstance(dataframes, (list, set, tuple)):
        raise TypeError("'dataframes' must be an iterable object")
    else:
        dataframes = list(dataframes)
    if isinstance(columns_to_join, str):
        columns_to_join = [columns_to_join]
    if common_columns == None:
        common_columns = set(dataframes[0].columns).intersection(
            *[set(dataframe.columns) for dataframe in dataframes[1:]]
        )
        for col in columns_to_join:
            common_columns.remove(col)
    if not isinstance(common_columns, list):
        common_columns = list(common_columns)
    if not len(common_columns):
        raise ValueError("no columns to merge on")
    else:
        cm = set(common_columns).intersection(columns_to_join)
        if len(cm) > 1:
            raise ValueError(f"{cm} are both in 'common_columns' and 'columns_to_join'")
        elif len(cm) == 1:
            raise ValueError(
                f"{list(cm)[0]} is both in 'common_columns' and 'columns_to_join'"
            )

    if len(dataframes) == 2:
        df = dataframes[0].merge(dataframes[1], on=common_columns, how="outer")

        def aggregate(args):
            x, y = args
            if isinstance(x, str) and isinstance(y, str):
                return separator.join([x, y])
            elif isinstance(x, str):
                return x
            elif isinstance(y, str):
                return y

        for col in columns_to_join:
            df[col] = df[[f"{col}_x", f"{col}_y"]].agg(aggregate, axis=1)
            df.drop([f"{col}_x", f"{col}_y"], axis=1, inplace=True)

        return df.drop_duplicates(ignore_index=True)
    else:
        dataframes = [
            integrate_dataframes(
                dataframes[:2],
                common_columns=common_columns,
                columns_to_join=columns_to_join,
                separator=separator,
            )
        ] + dataframes[2:]
        return integrate_dataframes(
            dataframes,
            common_columns=common_columns,
            columns_to_join=columns_to_join,
            separator=separator,
        )


def get_version(filepath, marker=None, ref_date=None, retrieved_version=None):
    version = None
    if marker:
        try:
            with open(filepath, "r") as infile:
                nline = 0
                while not version and nline < 100:  # checks only in the first 100 lines
                    line = infile.readline()
                    if marker in line:
                        version = re.findall(
                            "[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]+\.[0-9]+", line
                        )[0]
                    nline += 1
        except UnicodeDecodeError:
            with gzip.open(filepath, "rt") as infile:
                version = ""
                nline = 0
                while not version and nline < 100:  # checks only in the first 100 lines
                    line = infile.readline()
                    if marker in line:
                        version = re.findall(
                            "[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]+\.[0-9]+", line
                        )[0]
                    nline += 1
    elif ref_date:
        version = ref_date.strftime("%Y-%m-%d")
    file_timestamp = (
        datetime.fromtimestamp(os.path.getmtime(filepath))
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%d, %H:%M:%S")
    )
    if version:
        return f"{version}    (accessed: {file_timestamp} UTC)"
        if retrieved_version:
            return f"{retrieved_version} {version}    (accessed: {file_timestamp} UTC)"
    elif retrieved_version:
        return f"{retrieved_version}    (accessed: {file_timestamp} UTC)"
    else:
        return f"accessed: {file_timestamp} UTC"


class Singleton(type):
    """
        Used as metaclass prevents to create multiple instances of the same class
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class File:
    """
    General class for retrieving a file for a specific database
    """

    def __init__(
        self,
        folder,
        url,
        censored_url=None,
        sep="\t",
        skiprows=None,
        names=None,
        usecols=None,
        dtype="string",
        update=True,
        version_marker=None,
        retrieved_version=None,
        rename_columns=None,
        query=None,
        final_columns=None,
        fix_function=None,
        custom_read_function=None,
        session=None,
        post=False,
        data_to_send=None,
        sequential_identifier=0,
        comment=None,
        index_col=None,
    ):
        self.__folder = folder
        self.__url = url
        self.__censored_url = censored_url
        self.__sequential_identifier = sequential_identifier
        if not update:
            try:
                (
                    self.__filepath,
                    self.__filename,
                    self.__version,
                    self.__updated,
                ) = self.retrieve_file_info()
            except Exception as err:
                if update == False:
                    log.warning(err)
                    log.warning(
                        "Encountered an error retrieving a local copy of the file. Trying to download it..."
                    )
                try:
                    (
                        self.__filepath,
                        self.__filename,
                        self.__version,
                        self.__updated,
                    ) = download(
                        url,
                        self.__folder,
                        update=update,
                        version_marker=version_marker,
                        retrieved_version=retrieved_version,
                        session=session,
                        post=post,
                        data_to_send=data_to_send,
                    )
                    self.save_file_info()
                except Exception as err:
                    log.error(err)
                    log.error("Unable to get the file")
                    raise
        else:  # update = True
            try:
                (
                    self.__filepath,
                    self.__filename,
                    self.__version,
                    self.__updated,
                ) = download(
                    url,
                    self.__folder,
                    update=update,
                    version_marker=version_marker,
                    retrieved_version=retrieved_version,
                    session=session,
                    post=post,
                    data_to_send=data_to_send,
                )
                self.save_file_info()
            except Exception as err:
                log.warning(err)
                log.warning(
                    "Encountered an error downloading the file. Trying to use a local copy of it"
                )
                try:
                    (
                        self.__filepath,
                        self.__filename,
                        self.__version,
                        self.__updated,
                    ) = self.retrieve_file_info()
                except Exception as err:
                    log.error(err)
                    log.error("Unable to get the file")
                    raise
        self.__sep = sep
        self.__skiprows = skiprows
        self.__names = names
        self.__usecols = usecols
        self.__dtype = dtype
        self.__version_marker = version_marker
        self.__rename_columns = rename_columns
        self.__query = query
        self.__final_columns = final_columns
        self.__fix_function = fix_function
        self.__custom_read_function = custom_read_function
        self.__comment = comment
        self.__index_col = index_col
        self.read()

    def read(self, first_time=True):
        if first_time:
            log.info(f"Reading data in {self.__filename} for {self.__folder}")
        else:
            log.info(f"Reading data in cleaned-{self.__filename} for {self.__folder}")
        if self.__custom_read_function:  # for handling specific not implemented formats
            self.__content = self.__custom_read_function(self.__filepath)
        elif self.__filename.endswith(".obo"):
            try:
                self.__content = fastobo.load(self.__filepath)
            except:
                if first_time and self.__fix_function:
                    self.__filepath = self.__fix_function(
                        self.__filepath
                    )  # hoping to never use it
                    self.read(
                        first_time=False
                    )  # tries to fix the file ones, if uncapable raises the error
                else:
                    raise
        else:  # if files are not handled by a custom function or obo it assumes they are csv readable by pandas
            self.__content = pd.read_csv(
                self.__filepath,
                sep=self.__sep,
                skiprows=self.__skiprows,
                names=self.__names,
                usecols=self.__usecols,
                dtype=self.__dtype,
                comment=self.__comment,
                index_col=self.__index_col,
            )
            if self.__rename_columns:
                self.__content = self.__content.rename(columns=self.__rename_columns)
            if self.__query:
                self.__content = self.__content.query(self.__query)
            if self.__final_columns:
                self.__content = self.__content[self.__final_columns]
            self.__content = self.__content.reset_index(drop=True)

    def save_file_info(self):
        log.debug("Saving source info in json")
        if not os.path.isdir("data/sources"):
            os.makedirs("data/sources")
        with open("data/sources/sources.json", "r+") as infofile:
            # get previous data
            sources_data = json.load(infofile)
            # add relevant data
            url = self.__url if not self.__censored_url else self.__censored_url
            sources_data[self.__folder]["files"].update(
                {self.__filename: {"URL": url, "version": self.__version}}
            )
            # reset file pointer/offset at start
            infofile.seek(0)
            # save
            json.dump(sources_data, infofile, indent=4)

    def retrieve_file_info(self):
        try:
            return self.__filepath, self.__filename, self.__version, self.__updated
        except AttributeError:
            log.debug("Retrieving info from 'sources.json'")
            with open("data/sources/sources.json", "r+") as infofile:
                # get previous data
                sources_data = json.load(infofile)
                # retrieve relevant data
                try:
                    filename = list(sources_data[self.__folder]["files"])[
                        self.__sequential_identifier
                    ]
                    version = sources_data[self.__folder]["files"][filename]["version"]
                    url = sources_data[self.__folder]["files"][filename]["URL"]
                    filepath = f"data/sources/{self.__folder}/{filename}"
                    updated = False
                    return filepath, filename, version, updated
                except IndexError:  # raised by sequential_identifier (not enought files saved in sources.json for the relative database)
                    log.warning(
                        "Unable to retrieve file info from 'sources.json' (no info about previous versions)"
                    )
                    raise IOError("File info retrieval failed")

    @property
    def filename(self):
        return self.__filename

    @property
    def filepath(self):
        return self.__filepath

    @property
    def version(self):
        return self.__version

    @property
    def updated(self):
        return self.__updated

    @property
    def content(self):
        return self.__content

    @content.setter
    def content(self, modified_content):
        self.__content = modified_content

    def __eq__(self, other):
        return self.filename == other.filename and self.version == other.version

    def __hash__(self):
        return hash((self.filename, self.version))

    def __str__(self):
        return self.__filename

    def __repr__(self):
        return f"file: {self.__filename}"

    def __call__(self):
        if (
            isinstance(self.__content, pd.DataFrame)
            and "source" not in self.__content.columns
        ):
            db = self.__content.copy()
            db.insert(len(self.__content.columns), "source", self.__folder)
            return db
        else:
            try:
                return self.__content.copy()
            except:
                return self.__content


class Database:
    """
    General class for database retrieval
    """

    def __init__(
        self,
        update=True,
        license=None,
        license_url=None,
        registration_required=False,
        session=None,
        requirements=None,
    ):
        log = logging.getLogger(f"databases:{self.__class__.__name__}")
        log.info(f"Initializing {self.__class__.__name__}...")
        self.__update = update
        self.__license = license
        self.__license_url = license_url
        self.__registration_required = registration_required
        self.__session = session
        self.__requirements = (
            requirements if isinstance(requirements, list) else [requirements]
        )
        self.check_requirements()
        self.save_database_info(
            self.__class__.__name__, license, license_url, registration_required
        )
        self.__files = set()
        self.__versions = {}

    def save_database_info(self, folder, license, license_url, registration_required):
        if os.path.isfile("data/sources/sources.json"):
            with open("data/sources/sources.json", "r+") as infofile:
                # get previous data
                sources_data = json.load(infofile)
                # add relevant data
                if folder not in sources_data:
                    sources_data[folder] = {
                        "files": {},
                        "license": license,
                        "license_url": license_url,
                        "registration_required": registration_required,
                    }
                # reset file pointer/offset at start
                infofile.seek(0)
                # save
                json.dump(sources_data, infofile, indent=4)
        else:
            if not os.path.isdir("data/sources"):
                os.makedirs("data/sources")
            with open("data/sources/sources.json", "w") as infofile:
                json.dump(
                    {
                        folder: {
                            "files": {},
                            "license": license,
                            "license_url": license_url,
                            "registration_required": registration_required,
                        }
                    },
                    infofile,
                    indent=4,
                )

    def _add_file(
        self,
        url,
        censored_url=None,
        sep="\t",
        skiprows=None,
        names=None,
        usecols=None,
        dtype="string",
        update=None,
        version_marker=None,
        retrieved_version=None,
        rename_columns=None,
        query=None,
        final_columns=None,
        fix_function=None,
        custom_read_function=None,
        session=None,
        post=False,
        data_to_send=None,
        comment=None,
        index_col=None,
    ):
        if self.__update != None:
            update = self.__update
        if session is None:
            session = self.__session
        tmp_file = File(
            self.__class__.__name__,
            url,
            censored_url=censored_url,
            sep=sep,
            skiprows=skiprows,
            names=names,
            usecols=usecols,
            dtype=dtype,
            update=update,
            version_marker=version_marker,
            retrieved_version=retrieved_version,
            rename_columns=rename_columns,
            query=query,
            final_columns=final_columns,
            fix_function=fix_function,
            custom_read_function=custom_read_function,
            session=session,
            post=post,
            data_to_send=data_to_send,
            sequential_identifier=len(self.__files),
            comment=comment,
            index_col=index_col,
        )
        # self.__files[tmp_file.filename] = tmp_file
        self.__files.add(tmp_file)
        self.__versions[tmp_file.filename] = tmp_file.version
        return tmp_file

    def check_requirements(self):
        try:
            if self.__requirements != [None]:
                for requirement in self.__requirements:
                    try:
                        log.debug(f"Loading requirement: {requirement.__name__}")
                        setattr(
                            self.__class__,
                            f"_{self.__class__.__name__}__{requirement.__name__}",
                            requirement(update=self.__update),
                        )
                        # beware! they are not properties
                        # they can be changed by mistake
                        # using property(lambda x: requirement(update=self._update)) works only for single requirements
                    except NameError:
                        raise RuntimeError(
                            f"{self.name} requires {requirement.__name__}"
                        )
        except AttributeError as err:  # if self doesn't have attribute __requirements
            raise err

    @property
    def files(self):
        return self.__files

    @property
    def version(self):
        vv = self.__versions.values()
        if len(set(vv)) == 1:
            return list(vv)[0]
        else:
            return self.__versions

    @property
    def updated(self):
        if self.__files:
            return any([file.updated for file in self.__files])

    @property
    def update(self):
        """
            Returns the input argument 'update'
            It should not be used, check the 'updated' property
        """
        return self.__update

    @update.setter
    def _update(self, value):
        """
            Changes the value of the input argument 'update'
            It should not be used
        """
        log.warning("You are changing the 'update' input argument, beware!")
        self.__update = value

    @property
    def license(self):
        if license:
            return self.__license
        else:
            return "NA"

    @property
    def license_url(self):
        if self.__license_url:
            return self.__license_url
        else:
            return "NA"

    @property
    def registration_required(self):
        return self.__registration_required

    @property
    def requirements(self):
        return self.__requirements

    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"database: {self.__class__.__name__}"

    def __call__(self):
        return self


def get_human_gene_ids():
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=Gene&term="9606"[Taxonomy ID]+AND+alive[property]+AND+genetype+protein+coding[Properties]+AND+alive[prop]&retmax=1000000'
    response = requests.get(url, allow_redirects=True)
    page = response.content
    soup = BeautifulSoup(page, "lxml")
    hsa_gene_ids = {int(id.text) for id in soup.findAll("id")}
    return sorted(hsa_gene_ids)


def tqdm4ray(ids, *args, **kwargs):
    from tqdm import tqdm

    def to_iterator(ids):
        while ids:
            done, ids = ray.wait(ids)
            yield ray.get(done[0])

    return tqdm(to_iterator(ids), *args, **kwargs)


def get_best_match(query, choices, score_cutoff=75):
    from thefuzz import process as thefuzzprocess

    best_match = thefuzzprocess.extractOne(query, choices, score_cutoff=score_cutoff)

    if best_match:
        return best_match[0]
    else:
        return best_match
