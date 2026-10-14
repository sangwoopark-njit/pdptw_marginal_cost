"""One width and one type size for every generated table.

Left alone, each table comes out as wide as its own columns happen to need and
its caption spans the full measure, so no two tables line up and a caption
floats above a table half its width.  This rewrites each table float so that:

  * the tabular fills a fixed fraction of the measure, TARGET, using
    \\extracolsep{\\fill} to spread the slack across the column gaps;
  * the caption sits in a minipage of that same width, so caption and table
    share both edges;
  * the body is set at one size, SIZE, for every table.

A minipage redefines \\textwidth to its own width, so `tabular*{\\textwidth}`
inside one fills it exactly and the caption above is set to the same measure.
Both therefore land on TARGET by construction rather than by a number that has
to be kept in step with the data.

This lives here, not in the build script, because make_tables.py owns the text
of the generated block: a transformation applied anywhere else is undone the
next time the tables are re-spliced, which is how the paper lost this
formatting once already.
"""
from __future__ import annotations
import re

TARGET = 0.80          # fraction of the measure every table is opened out to
#: labels whose table carries enough columns to want the whole measure
WIDE = {"tab:main": 1.00}
SIZE = r"\small"       # one type size for every table body
SIZES = (r"\tiny", r"\scriptsize", r"\footnotesize", r"\small",
         r"\normalsize", r"\large")


def _braced(s, i):
    """The balanced-brace group starting at s[i] == '{'; returns (body, end)."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == i or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def _one(inner):
    """Rewrite one table float's contents, or return None to leave it alone."""
    if r"\resizebox" in inner or r"\begin{tabular*}" in inner:
        return None                      # already scaled, or already done
    if inner.count(r"\begin{tabular}") != 1:
        return None                      # nested or absent: not ours to touch
    t0 = inner.find(r"\begin{tabular}")
    t1 = inner.rfind(r"\end{tabular}")
    tab, head = inner[t0:t1 + len(r"\end{tabular}")], inner[:t0]
    if inner[t1 + len(r"\end{tabular}"):].strip():
        return None

    c = head.find(r"\caption")
    lab = re.search(r"\\label\{[^}]*\}", head)
    if c < 0 or lab is None:
        return None
    cap, c_end = _braced(head, head.index("{", c))
    frac = WIDE.get(lab.group(0)[7:-1], TARGET)

    m = re.search(r"\\begin\{tabular\}\{@\{\}", tab)
    if m is None:
        return None
    stretched = (tab[:m.start()]
                 + r"\begin{tabular*}{\tabw}{@{\extracolsep{\fill}}"
                 + tab[m.end():]).replace(r"\end{tabular}", r"\end{tabular*}")

    # The fraction is a floor, not a fixed width.  The tabular is set once into a box
    # to find what its columns actually need; if that exceeds the target the
    # table keeps its own width, since \extracolsep{\fill} can spread a table
    # out to fill a box but cannot pull one in, and forcing it overflows.
    return (r"\begin{table}[t]\centering" + "\n"
            + r"\sbox{\tabbox}{" + SIZE + tab + "}%\n"
            + "\\setlength{\\tabw}{%.2f\\textwidth}%%\n" % frac
            + r"\ifdim\wd\tabbox>\tabw\setlength{\tabw}{\wd\tabbox}\fi" + "\n"
            + r"\begin{minipage}{\tabw}" + "\n"
            + "\\centering\n"
            + "\\caption{" + cap + "}\n"
            + lab.group(0) + "\n"
            + SIZE + stretched + "\n"
            + "\\end{minipage}\n"
            + r"\end{table}")


#: `table` and `table*` both, because the star means nothing in a single-column
#: class -- and leaving the starred ones out is exactly how three of them ended
#: up at a different width and type size from the rest.
FLOAT = re.compile(r"\\begin\{table\*?\}(\[[^\]]*\])?(.*?)\\end\{table\*?\}",
                   re.S)


def restyle(body, verbose=True):
    """Apply the shared width and size to every plain table float in `body`."""
    n = skip = 0

    def one(m):
        nonlocal n, skip
        inner = re.sub(r"^\s*\\centering", "", m.group(2))
        new = _one(inner)
        if new is None:
            skip += 1
            return m.group(0)
        n += 1
        return new

    out = FLOAT.sub(one, body)
    if verbose:
        print(f"  restyled {n} tables to {TARGET:.0%} of the measure at "
              f"{SIZE.lstrip(chr(92))}" + (f"; left {skip} alone" if skip else ""))
    return out
