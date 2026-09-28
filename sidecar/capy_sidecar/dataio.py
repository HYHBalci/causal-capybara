"""Import, profile, and health-check a dataset.

Typed import: date formats, value labels from Stata, survey weights, clustered
ids -- and they survive reopening the project, because the parquet keeps them.
"""

from __future__ import annotations

import codecs
import csv
import gzip
import hashlib
import io
import math
import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED = {
    ".csv": "Comma-separated text",
    ".tsv": "Tab-separated text",
    ".txt": "Delimited text",
    ".parquet": "Apache Parquet",
    ".pq": "Apache Parquet",
    ".dta": "Stata",
    ".sav": "SPSS",
    ".zsav": "SPSS (compressed)",
    ".por": "SPSS portable",
    ".sas7bdat": "SAS",
    ".xpt": "SAS transport",
    ".xlsx": "Excel",
    ".xlsm": "Excel",
    ".xls": "Excel (legacy)",
    ".feather": "Arrow / Feather",
    ".arrow": "Arrow IPC",
    ".json": "JSON records",
    ".gdt": "Gretl data (courtesy import)",
    ".gdtb": "Gretl binary data",
}


class ImportError_(Exception):
    """Raised with a sentence the import wizard can show."""


@dataclass
class Imported:
    df: pd.DataFrame
    columns: list[dict[str, Any]]
    options: dict[str, Any]
    notes: list[str]
    checksum: str


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

# The separators a spreadsheet is likely to have written. Order is only a
# tie-break: when two candidates explain the file exactly as well, the first one
# listed keeps it, and the comma is first because it is the commonest.
CANDIDATE_DELIMITERS = (",", ";", "\t", "|")

DELIMITER_NAMES = {",": "commas", ";": "semicolons", "\t": "tabs", "|": "vertical bars"}

# What each extension promises, so that only a surprise is worth telling the user about.
EXPECTED_DELIMITERS = {".csv": ",", ".tsv": "\t"}

# How many of the sampled lines a delimiter must split into the same number of
# fields before we believe it is really the separator.
MIN_DELIMITER_AGREEMENT = 0.75

# How each encoding is described to someone who has never heard of one. utf-8-sig
# reads ordinary UTF-8 as well and quietly drops the byte-order mark Excel likes to
# add; cp1252 is what "Save as CSV" writes on a Western Windows machine; latin-1 can
# never fail, so it is the last resort that at least gets the file open.
ENCODING_NAMES = {
    "utf-8": "UTF-8, the usual alphabet for data files",
    "utf-8-sig": "UTF-8, the usual alphabet for data files",
    "utf-16": "UTF-16, an alphabet that stores every letter in two bytes",
    "utf-16-le": "UTF-16, an alphabet that stores every letter in two bytes",
    "utf-16-be": "UTF-16, an alphabet that stores every letter in two bytes",
    "cp1252": "Windows-1252, the alphabet Excel writes on a Western Windows machine",
    "latin-1": "Latin-1, the older Western European alphabet",
}

TEXT_EXTENSIONS = (".csv", ".tsv", ".txt")

DEFAULT_NA_VALUES = ["", "NA", "N/A", "na", "NaN", ".", "#N/A", "null", "NULL"]

# Numbers some statistics packages write to mean "missing". We never blank them out
# behind the user's back -- that would be one more silent edit -- but we do say that
# we saw them, because an average of -412 is otherwise very hard to explain.
MISSING_CODES = (-999.0, -9999.0)

_RAGGED_ROW = re.compile(r"Expected (\d+) fields in line (\d+), saw (\d+)")
_MANGLED_NAME = re.compile(r"^(?P<base>.+)\.(?P<n>\d+)$")
_COMMA_DECIMAL = re.compile(r"^[-+]?\d+,\d+$")
_GROUPED_COMMA_DECIMAL = re.compile(r"^[-+]?\d{1,3}(?:\.\d{3})+,\d+$")
_DOT_DECIMAL = re.compile(r"^[-+]?\d+\.\d+$")
_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def checksum_file(path: Path, *, limit: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            read += len(chunk)
            if read >= limit:
                h.update(b"<truncated>")
                break
    return h.hexdigest()[:32]


def _one_line(exc: Exception) -> str:
    """A library's own words, squeezed onto one line and cut short, so that a
    message stays readable when we quote it."""
    text = " ".join(str(exc).split())
    return text[:197] + "..." if len(text) > 200 else text


@contextmanager
def _reading(path: Path, what: str, *, encoding: str | None = None) -> Iterator[None]:
    """Turn whatever a reader library throws into one sentence with a next step.

    pandas, pyarrow and pyreadstat each report failures in their own vocabulary, and
    every one of those messages used to reach the import screen unedited, so nothing
    may be left to escape this wrapper.
    """
    try:
        yield
    except ImportError_:
        raise
    except UnicodeDecodeError:
        named = ENCODING_NAMES.get(encoding or "", encoding or "the alphabet we expected")
        raise ImportError_(
            f"{path.name} is not written in {named}, so its text came out as nonsense. "
            'Open the file in your spreadsheet program and save it again as "CSV UTF-8", '
            "then import it once more. If it will not open there either, it is probably not "
            "a text file at all - check you picked the file you meant."
        ) from None
    except pd.errors.EmptyDataError:
        raise ImportError_(
            f"{path.name} is empty - there is nothing to read. Check you picked the right file."
        ) from None
    except pd.errors.ParserError as exc:
        # The reader says "Error tokenizing data. C error: Expected 2 fields in
        # line 3, saw 3", which names the broken row but in nobody's language.
        m = _RAGGED_ROW.search(str(exc))
        if m:
            raise ImportError_(
                f"Line {m.group(2)} of {path.name} holds {m.group(3)} values, but the file only "
                f"names {m.group(1)} columns, so its rows do not line up. Open the file at that "
                "line: usually one value contains the separator character and needs quotation "
                "marks around it."
            ) from None
        raise ImportError_(
            f"The rows in {path.name} do not line up, so it could not be read: {_one_line(exc)}. "
            "That usually means one row carries more values than there are column names, or a "
            "quotation mark was left open. Open the file at the line named above and fix that row."
        ) from None
    except MemoryError:
        raise ImportError_(
            f"{path.name} is too large to fit in this computer's memory. Import a smaller extract "
            "of it - a few hundred thousand rows is plenty to learn on."
        ) from None
    except PermissionError:
        raise ImportError_(
            f"{path.name} could not be opened. Another program - very often Excel - still has the "
            "file open, or you do not have permission to read it. Close it there and try again."
        ) from None
    except Exception as exc:
        raise ImportError_(
            f"Causal Capybara could not read {path.name} as {what}. The reader reported: "
            f"{_one_line(exc)}. Check that the file still opens in the program that wrote it, and "
            "that it finished downloading or copying."
        ) from None


def _decodes(head: bytes, encoding: str, *, partial: bool) -> bool:
    """Whether a chunk of the file is readable in this encoding. A chunk cut in the
    middle of a letter is not evidence against the encoding, so a failure in the last
    few bytes of a truncated read does not count."""
    try:
        head.decode(encoding)
        return True
    except UnicodeDecodeError as exc:
        return partial and exc.start >= len(head) - 4
    except LookupError:
        return False


def detect_encoding(path: Path, *, limit: int = 256 * 1024) -> str:
    """Work out which alphabet a text file is written in.

    Excel on Windows saves CSV in the local code page rather than UTF-8, and a single
    accented name in such a file used to end the import with a message about byte
    0xef, so the encoding is worked out here rather than assumed.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(limit + 1)
    except OSError:
        return "utf-8-sig"
    partial = len(head) > limit
    head = head[:limit]
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"
    # UTF-16 without a marker leaves a zero byte beside every plain letter, which no
    # single-byte encoding ever produces in real text.
    if b"\x00" in head[:4096]:
        return "utf-16-le" if head[1:2] == b"\x00" else "utf-16-be"
    for encoding in ("utf-8-sig", "cp1252"):
        if _decodes(head, encoding, partial=partial):
            return encoding
    return "latin-1"


def _text_head(path: Path, encoding: str, *, limit: int = 64 * 1024) -> tuple[str, bool]:
    """The first characters of the file, and whether more of it follows."""
    with path.open("r", encoding=encoding, errors="replace", newline="") as fh:
        text = fh.read(limit + 1)
    return (text[:limit], True) if len(text) > limit else (text, False)


def _sample_rows(head: str, delimiter: str, *, truncated: bool, limit: int = 20) -> list[list[str]]:
    """Split the head of the file the way a CSV reader would, so that quoted fields
    and values containing the delimiter are counted correctly."""
    text = head
    if truncated:
        cut = text.rfind("\n")
        text = text[: cut + 1] if cut >= 0 else ""
    rows: list[list[str]] = []
    try:
        for row in csv.reader(io.StringIO(text), delimiter=delimiter):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    except csv.Error:
        return []
    return rows


def _delimiter_score(rows: list[list[str]]) -> tuple[float, int]:
    """How well a delimiter explains the file: how many of the sampled lines agree on
    a field count, and what that count is. A delimiter that leaves one field per line
    has explained nothing, and scores zero."""
    if not rows:
        return (0.0, 0)
    counts = [len(r) for r in rows]
    modal = max(set(counts), key=counts.count)
    if modal < 2:
        return (0.0, modal)
    return (counts.count(modal) / len(counts), modal)


def _as_float(s: pd.Series) -> np.ndarray:
    """A column's values as plain floats, however pandas chose to store them.

    A nullable column (Int64, boolean, string) carries its own marker for a missing
    value, and handing that straight to numpy is an error on pandas 2, so the marker
    is spelled out here and every caller gets an ordinary array of floats.
    """
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _looks_numeric(token: str, decimal: str) -> bool:
    text = token.strip()
    if not text:
        return False
    text = text.replace(".", "").replace(",", ".") if decimal == "," else text.replace(",", "")
    try:
        float(text)
    except ValueError:
        return False
    return True


def sniff_delimiter(path: Path, *, encoding: str = "utf-8") -> str:
    """Which character separates the values in a text data file."""
    return str(_sniff_dialect(path, encoding)["delimiter"])


def _sniff_dialect(path: Path, encoding: str, *, delimiter: str | None = None) -> dict[str, Any]:
    """Work out how a text file is laid out by parsing its first lines.

    Counting characters, which is what this used to do, is fooled by every decimal
    comma in a European spreadsheet export: `y;x` with values like `1,5` holds more
    commas than semicolons, so the whole table collapsed into one column named `y;x`.
    Judging a candidate by whether it splits every line into the same plural number
    of fields cannot be fooled that way.
    """
    head, truncated = _text_head(path, encoding)
    if delimiter is None:
        best, best_score = ",", (0.0, 0)
        for candidate in CANDIDATE_DELIMITERS:
            score = _delimiter_score(_sample_rows(head, candidate, truncated=truncated))
            # A candidate that only explains some of the lines -- the header saying
            # one column while the rows say two, say -- is a comma inside somebody's
            # sentence, not the separator, so the ordinary comma is kept instead.
            # Between two candidates that do explain every line, the one that finds
            # more columns wins: it has accounted for more of what is written.
            if score[0] >= MIN_DELIMITER_AGREEMENT and score > best_score:
                best, best_score = candidate, score
        delimiter = best
    rows = _sample_rows(head, delimiter, truncated=truncated)

    # A comma left inside the fields of a semicolon- or tab-separated file is a
    # decimal point, not a separator: that is exactly how German, French and most
    # other European spreadsheets write 1,5 for one and a half.
    decimal, thousands = ".", None
    if delimiter != "," and len(rows) > 1:
        comma_decimals = grouped = dot_decimals = 0
        for row in rows[1:]:
            for cell in row:
                text = cell.strip()
                if _GROUPED_COMMA_DECIMAL.match(text):
                    grouped += 1
                    comma_decimals += 1
                elif _COMMA_DECIMAL.match(text):
                    comma_decimals += 1
                elif _DOT_DECIMAL.match(text):
                    dot_decimals += 1
        if comma_decimals and comma_decimals >= dot_decimals:
            decimal = ","
            thousands = "." if grouped else None

    # A first line made entirely of numbers is data, not names: reading it as a
    # header would eat a row of the data and label the columns with measurements.
    has_header = not (rows and all(_looks_numeric(cell, decimal) for cell in rows[0]))
    return {
        "delimiter": delimiter,
        "decimal": decimal,
        "thousands": thousands,
        "has_header": has_header,
        "header_row": [c.strip() for c in rows[0]] if (rows and has_header) else None,
    }


def _estimate_rows(path: Path, *, has_header: bool) -> tuple[int, bool]:
    """About how many data rows a text file holds, and whether that is an exact
    count. The preview reads only the first handful of rows and still has to tell the
    user how big the file it is offering to import actually is."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(1024 * 1024)
    except OSError:
        return (0, False)
    lines = head.count(b"\n")
    if head and not head.endswith(b"\n"):
        lines += 1
    if size <= len(head):
        return (max(0, lines - (1 if has_header else 0)), True)
    if lines == 0:
        return (0, False)
    estimated = max(0, int(size / (len(head) / lines)) - (1 if has_header else 0))
    # The first megabyte is not a fair sample of a long file's line lengths, so the
    # figure is rounded to two digits: "about 130,000 rows" promises no more than it
    # can keep, where 125,507 would look counted.
    digits = len(str(estimated))
    step = 10 ** max(1, digits - 2)
    return (int(round(estimated / step) * step), False)


def _default_v_names(count: int) -> list[str]:
    return [f"v{i + 1}" for i in range(count)]


def _all_numeric_names(names: Any) -> bool:
    parsed = [str(n).strip() for n in names]
    return bool(parsed) and all(_looks_numeric(n, ".") for n in parsed)


def read_any(path: str | Path, options: dict[str, Any] | None = None) -> Imported:
    """Read a file into a DataFrame, keeping value labels and dates where the
    format carries them."""
    p = Path(path)
    if not p.exists():
        raise ImportError_(
            f"There is no file at {p}. Check the spelling of the folder and file names, and that "
            "the drive or shared folder holding it is connected."
        )
    if p.is_dir():
        raise ImportError_(
            f"{p} is a folder, not a data file. Open the folder and pick the file inside it that "
            "holds your data."
        )
    try:
        empty = p.stat().st_size == 0
    except OSError:
        empty = False
    if empty:
        raise ImportError_(
            f"{p.name} is empty - there is nothing to read. Check you picked the right file."
        )
    opts = dict(options or {})
    ext = p.suffix.lower()
    notes: list[str] = []
    labels: dict[str, dict[Any, str]] = {}
    raw_names: list[str] | None = None

    if ext in TEXT_EXTENSIONS:
        # Even looking at the first lines can fail -- on a file Excel still has open,
        # for one -- and that has to arrive as a sentence too.
        with _reading(p, "a text data file"):
            encoding = opts.get("encoding") or detect_encoding(p)
            forced = opts.get("delimiter") or ("\t" if ext == ".tsv" else None)
            dialect = _sniff_dialect(p, encoding, delimiter=forced)
        sep = str(dialect["delimiter"])
        decimal = opts.get("decimal") or str(dialect["decimal"])
        thousands = opts.get("thousands") or dialect["thousands"]
        has_header = bool(dialect["has_header"]) if opts.get("header") is None else bool(opts["header"])
        with _reading(p, "a text data file", encoding=encoding):
            df = pd.read_csv(
                p,
                sep=sep,
                header=0 if has_header else None,
                na_values=opts.get("na_values") or DEFAULT_NA_VALUES,
                keep_default_na=True,
                encoding=encoding,
                engine="python" if len(sep) > 1 else "c",
                decimal=decimal,
                thousands=thousands,
                skiprows=opts.get("skiprows"),
                nrows=opts.get("nrows"),
            )
        if not has_header:
            df.columns = _default_v_names(df.shape[1])
            opened = (
                "The first line of this file already holds data rather than column names, so every "
                if opts.get("header") is None
                else "This file was read as having no column names, so every "
            )
            notes.append(
                opened + f"line was kept and the columns were named v1 to v{df.shape[1]}. Rename "
                "them from the variable list once you know what they hold."
            )
        elif not opts.get("skiprows"):
            raw_names = dialect["header_row"]
        if encoding not in ("utf-8", "utf-8-sig"):
            notes.append(
                "This file is not saved as UTF-8 text; it was read as "
                f"{ENCODING_NAMES.get(encoding, encoding)}. If any accented letters look wrong, "
                'save it again from your spreadsheet program as "CSV UTF-8" and import it again.'
            )
        # Saying "separated by commas" about a .csv file is noise, and so is saying
        # anything at all about a file with a single column, where nothing was
        # separated; saying it about a file that turned out to use semicolons and
        # decimal commas is the difference between a real table and nonsense.
        if df.shape[1] > 1 and (sep != EXPECTED_DELIMITERS.get(ext) or decimal != "."):
            how = f"Values in this file are separated by {DELIMITER_NAMES.get(sep, repr(sep))}"
            if decimal == ",":
                how += ", and numbers use a comma for the decimal point, so 1,5 was read as one and a half"
                if thousands == ".":
                    how += ", with a dot grouping thousands, so 1.234,50 was read as 1234.5"
            notes.append(how + ". That was worked out from the file itself.")
        opts["encoding"] = encoding
        opts["delimiter"] = sep
        opts["decimal"] = decimal
        opts["thousands"] = thousands
        opts["header"] = has_header
    elif ext in (".parquet", ".pq"):
        with _reading(p, "a Parquet file"):
            df = pd.read_parquet(p)
    elif ext == ".dta":
        with _reading(p, "a Stata file"):
            with pd.io.stata.StataReader(p, convert_categoricals=False) as reader:
                df = reader.read()
                try:
                    vl = reader.value_labels()
                    lbl_map = reader.lbllist
                    varlist = reader.varlist
                    for var, lname in zip(varlist, lbl_map):
                        if lname and lname in vl:
                            labels[var] = {k: str(v) for k, v in vl[lname].items()}
                except Exception:
                    pass
                try:
                    var_labels = reader.variable_labels()
                except Exception:
                    var_labels = {}
            if var_labels:
                opts["variable_labels"] = {k: v for k, v in var_labels.items() if v}
            if labels:
                notes.append(f"Kept Stata value labels for {len(labels)} variable(s).")
    elif ext in (".sav", ".zsav", ".por"):
        try:
            import pyreadstat
        except ImportError:
            raise ImportError_(
                "Causal Capybara cannot open SPSS files on this computer, because the part that "
                "reads them was not installed. Open the file in SPSS and save it as CSV, then "
                "import that instead."
            ) from None
        with _reading(p, "an SPSS file"):
            df, meta = pyreadstat.read_sav(str(p)) if ext != ".por" else pyreadstat.read_por(str(p))
        labels = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in (meta.variable_value_labels or {}).items()}
        if meta.column_names_to_labels:
            opts["variable_labels"] = {k: v for k, v in meta.column_names_to_labels.items() if v}
    elif ext in (".sas7bdat", ".xpt"):
        with _reading(p, "a SAS file"):
            df = pd.read_sas(p, encoding="latin-1")
    elif ext in (".xlsx", ".xlsm", ".xls"):
        df = _read_excel(p, opts, notes)
    elif ext in (".feather", ".arrow"):
        with _reading(p, "an Arrow file"):
            df = pd.read_feather(p)
    elif ext == ".json":
        with _reading(p, "a JSON file"):
            df = pd.read_json(p, orient=opts.get("orient", "records"))
    elif ext in (".gdt", ".gdtb"):
        with _reading(p, "a Gretl workfile"):
            df, gnotes, labels = read_gretl(p)
        notes.extend(gnotes)
    elif not ext:
        raise ImportError_(
            f"{p.name} has no file extension, so there is no way to tell what is inside it. Rename "
            "it so that it ends in .csv if it is comma-separated text, and import it again."
        )
    else:
        raise ImportError_(
            f"Causal Capybara does not read '{ext}' files yet. "
            f"Supported: {', '.join(sorted(SUPPORTED))}."
        )

    if df is None or df.shape[1] == 0:
        raise ImportError_(
            f"{p.name} was opened but holds no columns. Check it is the file you meant to pick, and "
            "that it has a row of column names at the top."
        )

    df = normalise_columns(df, notes, raw_names=raw_names)
    df = coerce_dates(df, opts, notes)
    notes.extend(_value_notes(df))
    if len(df) == 0:
        notes.append(
            "This file has column names but no data rows, so there is nothing to analyse yet. The "
            "export that produced it was probably empty - check it and import it again."
        )
    cols = profile_columns(df, labels=labels, variable_labels=opts.get("variable_labels") or {})
    if labels:
        opts["value_labels"] = {k: {str(kk): vv for kk, vv in v.items()} for k, v in labels.items()}
    with _reading(p, "a data file"):
        checksum = checksum_file(p)
    return Imported(df=df, columns=cols, options=opts, notes=notes, checksum=checksum)


def _read_excel(path: Path, opts: dict[str, Any], notes: list[str]) -> pd.DataFrame:
    """Read one sheet of a workbook, and say which sheets were left behind.

    Real workbooks nearly always open on a cover or notes sheet, so importing the
    first one without a word is how a user ends up analysing a read-me.
    """
    with _reading(path, "an Excel workbook"):
        try:
            book = pd.ExcelFile(path)
        except ImportError:
            raise ImportError_(
                "Causal Capybara cannot open Excel files on this computer, because the part that "
                'reads them was not installed. Open the workbook in Excel, use "Save as" to write '
                "a CSV file, and import that instead."
            ) from None
        except Exception:
            raise ImportError_(
                f"{path.name} could not be opened as an Excel workbook. It may be damaged, or it "
                "may be another kind of file that was renamed. Try opening it in Excel: if it "
                'opens there, use "Save as" to write it again as .xlsx or as CSV.'
            ) from None
        with book:
            sheets = [str(s) for s in book.sheet_names]
            if not sheets:
                raise ImportError_(
                    f"{path.name} holds no sheets at all. Check you picked the right file."
                )
            wanted = opts.get("sheet")
            if wanted is None or wanted == "":
                chosen = sheets[0]
            elif isinstance(wanted, int) and not isinstance(wanted, bool):
                if not 0 <= wanted < len(sheets):
                    raise ImportError_(
                        f"{path.name} has {len(sheets)} sheet(s), so there is no sheet number "
                        f"{wanted}. Its sheets are: {', '.join(sheets)}."
                    )
                chosen = sheets[wanted]
            elif str(wanted) in sheets:
                chosen = str(wanted)
            else:
                raise ImportError_(
                    f"{path.name} has no sheet called '{wanted}'. Its sheets are: "
                    f"{', '.join(sheets)}."
                )
            skiprows = opts.get("skiprows")
            asked = opts.get("header")
            df = book.parse(chosen, skiprows=skiprows)
            # A row of numbers is not a row of names; but if the user has said which
            # it is, that answer is the one that counts.
            has_header = not _all_numeric_names(df.columns) if asked is None else bool(asked)
            if not has_header:
                df = book.parse(chosen, skiprows=skiprows, header=None)
                df.columns = _default_v_names(df.shape[1])
                opened = (
                    "The first row of this sheet already holds data rather than column names, so "
                    if asked is None
                    else "This sheet was read as having no column names, so "
                )
                notes.append(
                    opened + f"every row was kept and the columns were named v1 to v{df.shape[1]}. "
                    "Rename them from the variable list once you know what they hold."
                )
    opts["sheet"] = chosen
    opts["sheets"] = sheets
    opts["header"] = has_header
    if len(sheets) > 1 and (wanted is None or wanted == ""):
        others = _join_names([s for s in sheets if s != chosen])
        notes.append(
            f"This workbook holds {len(sheets)} sheets. '{chosen}' was imported because it comes "
            f"first; the others are {others}. If that is not the table you meant, choose another "
            "sheet and import again."
        )
    return df


def preview(path: str | Path, options: dict[str, Any] | None = None, limit: int = 50) -> dict[str, Any]:
    """Read the first few rows of a file so the import screen can show what it is
    about to do, before anything is written into the project.

    Pointing at a file that turns out to be broken is an ordinary thing for a person
    to do, so this never raises: a failure comes back as {"error": one sentence} to
    be shown beside the file they picked.
    """
    # Even the row limit arrives from outside, so it cannot be trusted to be a
    # number; and a caller asking for a million rows would not get a preview.
    try:
        rows_wanted = max(1, min(1000, int(limit)))
    except (TypeError, ValueError):
        rows_wanted = 50
    try:
        p = Path(path)
        opts = dict(options or {})
        opts.setdefault("nrows", rows_wanted)
        imported = read_any(p, opts)
        used = dict(imported.options)
        # The preview reads a slice, so its row limit must not travel on into the
        # real import as though the user had asked for it.
        used.pop("nrows", None)
        frame = imported.df.head(rows_wanted)
        columns = [
            {"name": str(c), "dtype": str(frame[c].dtype), "kind": infer_kind(frame[c])}
            for c in frame.columns
        ]
        rows = [[_plain(v) for v in row] for row in frame.itertuples(index=False, name=None)]
        ext = p.suffix.lower()
        if ext in TEXT_EXTENSIONS and len(imported.df) >= rows_wanted:
            n_rows, exact = _estimate_rows(p, has_header=bool(used.get("header", True)))
        else:
            n_rows, exact = len(imported.df), True
        detected = {
            "format": SUPPORTED.get(ext, "data file"),
            "encoding": used.get("encoding"),
            "delimiter": used.get("delimiter"),
            "decimal": used.get("decimal"),
            "thousands": used.get("thousands"),
            "header": used.get("header"),
            "sheet": used.get("sheet"),
            "sheets": used.get("sheets") or [],
            "date_columns": used.get("date_columns") or [],
        }
        return {
            "columns": columns,
            "rows": rows,
            "notes": list(imported.notes),
            "options": used,
            "detected": detected,
            "n_rows_estimate": int(n_rows),
            "n_rows_exact": bool(exact),
        }
    except ImportError_ as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {
            "error": (
                f"Causal Capybara could not read that file: {_one_line(exc)}. Check that it opens "
                "in the program that wrote it, and that it finished copying."
            )
        }


def _join_names(names: list[str]) -> str:
    """A list of names as a person would say it: 'a', 'b' and 'c'."""
    quoted = [f"'{n}'" for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _unmangle(names: list[str]) -> list[str]:
    """Undo the reader's own de-duplication, which turns a second 'score' column into
    'score.1' before this module ever sees the frame, so that the rename can be
    reported instead of appearing from nowhere in the variable list."""
    present = set(names)
    out: list[str] = []
    for name in names:
        m = _MANGLED_NAME.match(name)
        out.append(m.group("base") if m and m.group("base") in present else name)
    return out


def _placeholder(index: int, taken: set[str]) -> str:
    """A name for a column the file left unnamed. It steps past any name already in
    the file, so that inventing a name for column 2 cannot quietly push a real
    column called v2 aside and rename it."""
    n = index + 1
    while f"v{n}" in taken:
        n += 1
    return f"v{n}"


def normalise_columns(
    df: pd.DataFrame, notes: list[str], *, raw_names: list[str] | None = None
) -> pd.DataFrame:
    """Unique, non-empty column names. Never silently reorder or drop."""
    seen: dict[str, int] = {}
    out: list[str] = []
    blanks: list[str] = []
    was_blank: set[int] = set()
    taken = {str(c).strip() for c in df.columns if str(c).strip()}
    for i, c in enumerate(df.columns):
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = _placeholder(i, taken)
            taken.add(name)
            blanks.append(name)
            was_blank.add(i)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    if blanks:
        blank_note = (
            f"{len(blanks)} column(s) had no name in the file and are now called "
            f"{_join_names(blanks)}. Rename them from the variable list once you know what they hold."
        )
        # A title banner merged across the top of a spreadsheet leaves every cell of
        # the header row empty but the first, and the real names then sit in row one
        # being read as data.
        if len(blanks) >= max(2, len(out) - 1):
            blank_note += (
                " If the top row of your file is a title rather than column names, skip that row "
                "in the import step and import the file again."
            )
        notes.append(blank_note)

    # The names the file itself used, so that a repeated one can be named in the
    # note. Columns that had no name at all are left out: several of those are not a
    # repeated name, and they have been reported already.
    original = [str(n).strip() for n in raw_names] if raw_names and len(raw_names) == len(out) else _unmangle(out)
    groups: dict[str, list[str]] = {}
    for i, (was, now) in enumerate(zip(original, out)):
        if i not in was_blank:
            groups.setdefault(was, []).append(now)
    for was, now in groups.items():
        if len(now) > 1:
            notes.append(
                f"{len(now)} columns in the file were called '{was}'. They are now "
                f"{_join_names(now)}, in the order they appear in the file, so that they can be "
                "told apart - check which one you mean before you use it."
            )
    df = df.copy()
    df.columns = out
    return df


def _value_notes(df: pd.DataFrame) -> list[str]:
    """Say out loud the two things a numeric column can hide: values no arithmetic
    can use, and the sentinel numbers other packages write to mean 'missing'."""
    notes: list[str] = []
    for col in df.columns:
        s = df[col]
        if s.empty or not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        arr = _as_float(s)
        n_inf = int(np.isinf(arr).sum())
        if n_inf:
            notes.append(
                f"'{col}' holds {n_inf} value(s) that are infinite rather than a number. No summary "
                "or estimate can use them, and they show as blank cells in the data sheet."
            )
        for code in MISSING_CODES:
            hits = int(np.count_nonzero(arr == code))
            if hits:
                notes.append(
                    f"'{col}' holds {hits} value(s) of {code:g}, which some statistics programs "
                    "write to mean 'missing'. They were kept as ordinary numbers - if they do mean "
                    "missing here, they will drag the average down until you replace them."
                )
    return notes


DATE_HINTS = ("date", "time", "day", "month", "year", "period", "dt", "week", "quarter")


def _date_order(sample: pd.Series) -> tuple[bool, bool]:
    """Read a day/month/year column the way it was written: return whether the day
    comes first, and whether that had to be guessed.

    01/02/2020 is the second of January in the United States and the first of
    February nearly everywhere else, and the file decides the answer for us only when
    one of its values carries a component above 12.
    """
    firsts: list[int] = []
    seconds: list[int] = []
    for value in sample:
        m = _NUMERIC_DATE.match(str(value))
        if m:
            firsts.append(int(m.group(1)))
            seconds.append(int(m.group(2)))
    if any(f > 12 for f in firsts):
        return True, False
    if any(s > 12 for s in seconds):
        return False, False
    if not firsts:
        return False, False
    return False, True


def coerce_dates(df: pd.DataFrame, opts: dict[str, Any], notes: list[str]) -> pd.DataFrame:
    """Parse obvious date columns; a 4-digit year stays an integer on purpose."""
    explicit = opts.get("date_columns") or []
    fmt = opts.get("date_format")
    dayfirst_opt = opts.get("dayfirst")
    converted: list[str] = []
    for col in df.columns:
        s = df[col]
        want = col in explicit
        if not want:
            # Only something written as text can be a date. A number is never guessed
            # at: a column called `time` holding -1.5 was being turned into a
            # timestamp in 1969, which destroyed the very column a panel is built on.
            if (
                pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)
            ):
                continue
            if not any(h in col.lower() for h in DATE_HINTS):
                continue
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            continue
        if not want and not sample.str.contains(r"[-/:]").mean() > 0.8:
            continue
        if dayfirst_opt is None:
            dayfirst, guessed = _date_order(sample)
        else:
            dayfirst, guessed = bool(dayfirst_opt), False
        parsed = _to_datetime(s, fmt=fmt, dayfirst=dayfirst)
        if parsed is None or not parsed.notna().mean() > 0.9:
            # Timestamps written for different UTC offsets cannot share one column
            # unless they are all put on the same clock first.
            retry = _to_datetime(s, fmt=fmt, dayfirst=dayfirst, utc=True)
            if retry is not None and retry.notna().mean() > 0.9:
                parsed = retry
                notes.append(
                    f"'{col}' holds times written for different time zones, so they were all put on "
                    "UTC (the world clock) to make them comparable."
                )
            else:
                if want:
                    notes.append(
                        f"'{col}' was asked for as dates but could not be read as dates, so it was "
                        "left as text. Check that its values are all written the same way round."
                    )
                continue
        df[col] = parsed
        converted.append(col)
        opts["dayfirst"] = dayfirst
        shown = parsed.dropna()
        if guessed and not shown.empty:
            ts = shown.iloc[0]
            notes.append(
                f"Dates in '{col}' could be read either way round: '{sample.iloc[0]}' was taken as "
                f"{ts.day} {MONTH_NAMES[ts.month - 1]} {ts.year} - month first, which is what a "
                "spreadsheet does. If your dates put the day first, say so and import again."
            )
    if converted:
        notes.append(f"Parsed as dates: {', '.join(converted)}.")
        opts["date_columns"] = converted
    return df


def _to_datetime(s: pd.Series, *, fmt: str | None, dayfirst: bool, utc: bool = False) -> pd.Series | None:
    """pandas raises for some inputs and merely warns about others, and neither is a
    reason to abandon an import, so a failure comes back as None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(s, format=fmt, dayfirst=dayfirst, errors="coerce", utc=utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gretl .gdt workfile import
# ---------------------------------------------------------------------------


def read_gretl(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, dict[Any, str]]]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, count=1)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ImportError_(f"This does not look like a Gretl .gdt file: {exc}") from None

    notes: list[str] = []
    var_el = root.find("variables")
    if var_el is None:
        raise ImportError_("The .gdt file has no <variables> block.")
    names: list[str] = []
    descriptions: dict[str, str] = {}
    for v in var_el.findall("variable"):
        name = v.get("name") or f"v{len(names) + 1}"
        names.append(name)
        if v.get("label"):
            descriptions[name] = v.get("label", "")

    obs_el = root.find("observations")
    if obs_el is None:
        raise ImportError_("The .gdt file has no <observations> block.")
    has_labels = (obs_el.get("labels") or "false").lower() in ("true", "1")
    rows: list[list[Any]] = []
    obs_labels: list[str] = []
    for obs in obs_el.findall("obs"):
        if has_labels and obs.get("label"):
            obs_labels.append(obs.get("label", ""))
        parts = (obs.text or "").split()
        vals: list[Any] = []
        for token in parts:
            if token in ("NA", "na", "."):
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(token))
                except ValueError:
                    vals.append(np.nan)
        rows.append(vals)

    width = len(names)
    rows = [r[:width] + [np.nan] * max(0, width - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=names)

    freq = root.get("frequency")
    startobs = root.get("startobs")
    dtype = root.get("type", "cross-section")
    if dtype == "time-series" and startobs and freq:
        try:
            df.insert(0, "obs_period", _gretl_time_index(startobs, int(freq), len(df)))
            notes.append(f"Gretl time series ({startobs}, frequency {freq}) added as 'obs_period'.")
        except Exception:
            pass
    if obs_labels and len(obs_labels) == len(df):
        df.insert(0, "obs_label", obs_labels)
    desc = root.find("description")
    if desc is not None and (desc.text or "").strip():
        notes.append("Gretl description: " + " ".join((desc.text or "").split())[:200])
    notes.append(f"Imported {len(df)} observations on {width} variables from a Gretl workfile.")
    return df, notes, {}


def _gretl_time_index(startobs: str, freq: int, n: int) -> list[float]:
    if ":" in startobs or "." in startobs:
        sep = ":" if ":" in startobs else "."
        y, s = startobs.split(sep)
        year, sub = int(y), int(s)
    else:
        year, sub = int(float(startobs)), 1
    out = []
    for _ in range(n):
        out.append(year + (sub - 1) / max(freq, 1))
        sub += 1
        if sub > freq:
            sub = 1
            year += 1
    return out


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def infer_kind(series: pd.Series) -> str:
    n = len(series)
    nun = int(series.nunique(dropna=True))
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == bool:
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        if nun <= 2:
            return "binary"
        if pd.api.types.is_integer_dtype(series) and n > 20 and nun > 0.9 * n:
            return "id"
        if pd.api.types.is_integer_dtype(series) and nun <= 12:
            return "categorical"
        return "continuous"
    if nun <= 2:
        return "binary"
    if n > 20 and nun > 0.5 * n:
        try:
            mean_len = float(series.dropna().astype(str).str.len().mean())
        except Exception:
            mean_len = 0.0
        return "text" if mean_len > 24 else "id"
    return "categorical"


def profile_columns(
    df: pd.DataFrame,
    *,
    labels: dict[str, dict[Any, str]] | None = None,
    variable_labels: dict[str, str] | None = None,
    sample_rows: int = 200_000,
) -> list[dict[str, Any]]:
    labels = labels or {}
    variable_labels = variable_labels or {}
    work = df if len(df) <= sample_rows else df.sample(sample_rows, random_state=0)
    out: list[dict[str, Any]] = []
    for col in df.columns:
        s = work[col]
        kind = infer_kind(s)
        entry: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "kind": kind,
            "label": variable_labels.get(col),
            "n_missing": int(df[col].isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            # A table with no rows has no percentage missing; saying 0.0 keeps a
            # number in front of the user instead of a blank where one should be.
            "pct_missing": round(float(df[col].isna().mean()) * 100, 2) if len(df) else 0.0,
        }
        if labels.get(col):
            entry["labels"] = {str(k): v for k, v in labels[col].items()}
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            arr = _as_float(s)
            # Infinities are not a range: left in, they turn every statistic below
            # into a blank, which the inspector then shows as "Range 0 ... 0".
            n_infinite = int(np.isinf(arr).sum())
            if n_infinite:
                entry["n_infinite"] = n_infinite
            arr = arr[np.isfinite(arr)]
            if arr.size:
                q = np.quantile(arr, [0, 0.25, 0.5, 0.75, 1.0])
                entry.update(
                    {
                        "min": _plain(q[0]), "q1": _plain(q[1]), "median": _plain(q[2]),
                        "q3": _plain(q[3]), "max": _plain(q[4]),
                        "mean": _plain(np.mean(arr)),
                        "sd": _plain(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                    }
                )
        elif pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            entry["min"] = str(s.min())
            entry["max"] = str(s.max())
        if kind in ("categorical", "binary") and entry["n_unique"] <= 40:
            vc = s.value_counts(dropna=True).head(40)
            entry["levels"] = [
                {"value": _plain(k), "count": int(v),
                 "label": (labels.get(col) or {}).get(k) or (labels.get(col) or {}).get(str(k))}
                for k, v in vc.items()
            ]
        out.append(entry)
    return out


def _plain(v: Any) -> Any:
    """A value that YAML and JSON can both write. Anything not finite becomes None,
    because a NaN or an infinity survives as far as the browser and then reads there
    as an ordinary missing value."""
    # pandas has three ways of saying "missing": None, NaT, and pd.NA for its
    # nullable columns. All three have to become a blank cell; pd.NA left alone
    # reaches the screen as the literal text "<NA>".
    if v is None or v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if math.isfinite(v) else None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    # A numpy string is a str, but the plain YAML writer refuses to write one, so it
    # has to be copied into a real str before it can be saved.
    if isinstance(v, str):
        return str(v)
    return v if isinstance(v, (int, bool)) or v is None else str(v)


def column_distribution(df: pd.DataFrame, column: str, *, bins: int = 24) -> dict[str, Any]:
    """The little chart the inspector shows when you click a variable."""
    if column not in df.columns:
        raise ImportError_(f"No variable called '{column}'.")
    s = df[column]
    kind = infer_kind(s)
    if kind in ("continuous",) and pd.api.types.is_numeric_dtype(s):
        arr = _as_float(s)
        # A single infinity makes the whole histogram impossible to draw, so the
        # values that cannot sit on an axis are left out of the picture.
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"kind": kind, "rows": []}
        counts, edges = np.histogram(arr, bins=min(bins, max(5, int(np.sqrt(arr.size)))))
        return {
            "kind": "histogram",
            "rows": [
                {"x_lo": float(edges[i]), "x_hi": float(edges[i + 1]),
                 "x": float((edges[i] + edges[i + 1]) / 2), "count": int(counts[i])}
                for i in range(len(counts))
            ],
        }
    if pd.api.types.is_datetime64_any_dtype(s):
        g = s.dropna().dt.to_period("M").value_counts().sort_index()
        return {"kind": "bar", "rows": [{"label": str(k), "value": int(v)} for k, v in g.items()][:200]}
    vc = s.value_counts(dropna=False).head(40)
    return {
        "kind": "bar",
        "rows": [{"label": "(missing)" if pd.isna(k) else str(k), "value": int(v)} for k, v in vc.items()],
    }


# ---------------------------------------------------------------------------
# Data health -- how a policy user notices they do not have a panel
# ---------------------------------------------------------------------------


def data_health(df: pd.DataFrame, roles: dict[str, Any] | None = None, *, max_cells: int = 6000) -> dict[str, Any]:
    roles = roles or {}
    unit = roles.get("unit")
    time = roles.get("time")
    treat = roles.get("treatment")
    out: dict[str, Any] = {"n": int(len(df)), "p": int(df.shape[1]), "warnings": []}
    if len(df) == 0:
        # Without this the panel reports "0 of 0 rows are complete (0%)" as though
        # that were an ordinary reading of a healthy table.
        out["warnings"].append(
            {
                "level": "warning",
                "message": "This table has column names but no rows, so nothing can be estimated "
                           "from it. Check the file you imported in your spreadsheet program - the "
                           "export that produced it was probably empty - and import it again.",
            }
        )

    miss = df.isna().mean() if len(df) else None
    out["missingness"] = [
        {
            "variable": c,
            "pct_missing": round(float(miss[c]) * 100, 2) if miss is not None else 0.0,
            "n_missing": int(df[c].isna().sum()),
        }
        for c in df.columns
    ]
    complete = int(df.notna().all(axis=1).sum())
    out["complete_cases"] = complete
    out["complete_case_pct"] = round(100.0 * complete / max(len(df), 1), 1)
    if complete < len(df):
        out["warnings"].append(
            {
                "level": "caution",
                "message": f"{len(df) - complete} of {len(df)} rows have at least one missing value. "
                           "A complete-case analysis would drop them, and the drop will be logged.",
            }
        )

    dup_all = int(df.duplicated().sum())
    if dup_all:
        out["warnings"].append({"level": "caution", "message": f"{dup_all} fully duplicated row(s)."})
    out["duplicate_rows"] = dup_all

    if unit and unit in df.columns:
        out["n_units"] = int(df[unit].nunique())
        if time and time in df.columns:
            out.update(_panel_health(df, unit, time, treat, max_cells=max_cells, warnings=out["warnings"]))
        else:
            dup_ids = int(df[unit].duplicated().sum())
            out["duplicate_unit_rows"] = dup_ids
            if dup_ids:
                out["warnings"].append(
                    {"level": "caution",
                     "message": f"{dup_ids} row(s) repeat a unit id with no time variable set. "
                                "If this is a panel, assign a time variable."}
                )
    if time and time in df.columns and not unit:
        out["n_periods"] = int(df[time].nunique())
    return out


def _panel_health(
    df: pd.DataFrame, unit: str, time: str, treat: str | None, *, max_cells: int, warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_units = int(df[unit].nunique())
    n_periods = int(df[time].nunique())
    out["n_units"] = n_units
    out["n_periods"] = n_periods
    per_unit = df.groupby(unit, observed=True)[time].nunique()
    out["balanced"] = bool(per_unit.nunique() == 1 and int(per_unit.iloc[0]) == n_periods)
    out["min_periods_per_unit"] = int(per_unit.min()) if n_units else 0
    out["max_periods_per_unit"] = int(per_unit.max()) if n_units else 0
    dup = int(df.duplicated(subset=[unit, time]).sum())
    out["duplicate_unit_time_rows"] = dup
    if dup:
        warnings.append(
            {"level": "warning",
             "message": f"{dup} row(s) repeat the same unit and time. Panel estimators need one row per "
                        "unit-period; fix the ids or aggregate first."}
        )
    if not out["balanced"]:
        warnings.append(
            {"level": "info",
             "message": f"Unbalanced panel: units are observed between {out['min_periods_per_unit']} and "
                        f"{out['max_periods_per_unit']} periods."}
        )

    # entry / exit
    times_sorted = sorted(pd.unique(pd.to_numeric(df[time], errors="coerce").dropna()))
    if times_sorted:
        first = df.groupby(unit, observed=True)[time].min()
        last = df.groupby(unit, observed=True)[time].max()
        out["entry_after_start"] = int((pd.to_numeric(first, errors="coerce") > times_sorted[0]).sum())
        out["exit_before_end"] = int((pd.to_numeric(last, errors="coerce") < times_sorted[-1]).sum())

    # treatment timing sketch: when does each unit turn on?
    if treat and treat in df.columns:
        try:
            from capy_py.stats import to01

            t = pd.Series(to01(df[treat]), index=df.index)
            work = df.assign(_t=t.to_numpy(), _time=pd.to_numeric(df[time], errors="coerce").to_numpy())
            on = work.loc[work["_t"] > 0.5]
            cohorts = on.groupby(unit, observed=True)["_time"].min()
            cohorts = cohorts.reindex(pd.Index(pd.unique(df[unit]))).fillna(0)
            counts = cohorts.value_counts().sort_index()
            out["adoption"] = [
                {"cohort": (None if float(k) == 0 else float(k)), "n_units": int(v)} for k, v in counts.items()
            ]
            treated_cohorts = [c for c in counts.index if float(c) != 0]
            out["n_never_treated"] = int(counts.get(0, 0))
            out["staggered"] = bool(len(treated_cohorts) > 1)
            if out["staggered"]:
                warnings.append(
                    {"level": "info",
                     "message": "Adoption is staggered: groups are treated at different times. "
                                "Two-way fixed effects is probably the wrong default here."}
                )
            if out["n_never_treated"] == 0 and treated_cohorts:
                warnings.append(
                    {"level": "caution",
                     "message": "Every unit is eventually treated. Staggered estimators will compare "
                                "not-yet-treated units instead of never-treated ones."}
                )
            # a small unit x time grid for the heatmap, downsampled to stay quick
            units = list(pd.unique(df[unit]))
            periods = times_sorted
            if units and periods:
                step_u = max(1, len(units) * len(periods) // max_cells)
                keep_units = units[::step_u][:200]
                grid = work.loc[work[unit].isin(keep_units), [unit, "_time", "_t"]]
                out["timing_grid"] = [
                    {"unit": str(r[0]), "time": float(r[1]), "value": float(r[2])}
                    for r in grid.itertuples(index=False, name=None)
                ][: max_cells]
        except Exception:
            pass
    return out


def missingness_by_time(df: pd.DataFrame, time: str, columns: list[str] | None = None,
                        max_cells: int = 4000) -> list[dict[str, Any]]:
    """The missingness heatmap: variable x period."""
    if time not in df.columns:
        return []
    cols = columns or [c for c in df.columns if c != time]
    periods = sorted(pd.unique(df[time].dropna()))
    if len(periods) * len(cols) > max_cells:
        step = max(1, (len(periods) * len(cols)) // max_cells)
        periods = periods[::step]
    rows = []
    for p in periods:
        sub = df.loc[df[time] == p]
        if sub.empty:
            continue
        for c in cols:
            rows.append({"time": str(p), "unit": c, "value": round(float(sub[c].isna().mean()) * 100, 2)})
    return rows
