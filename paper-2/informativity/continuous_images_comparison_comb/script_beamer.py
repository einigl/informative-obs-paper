import os
import sys
from typing import List
from datetime import date

script_path = os.path.dirname(__file__)
beamers_path = os.path.join(script_path, "beamers")

sys.path.insert(0, os.path.join(script_path, "..", "..", ".."))
sys.path.insert(1, os.path.join(script_path, "..", ".."))

from orion_util import latex_line, latex_param
from infobs.plots import Plotter

lines = [
    "13co10",
    "12co10",
    "c18o10",
    "32so21",
    "12cn10",
    "n2hp10",
    "cch10",
    "hcop10",
    "h2co",
    "12cs21",
    "hcn10",
    "hnc10"
]

param = sys.argv[1]
stat = sys.argv[2]

plotter = Plotter(
    line_formatter=latex_line,
    param_formatter=latex_param
)

if not os.path.isdir(beamers_path):
    os.mkdir(beamers_path)

def build_frame(filename: str, lines: List[str]):
    return rf"""
\begin§|frame|§§|Informativity on ${plotter.params_comb_formatter(param)}$ of ${plotter.lines_comb_formatter(lines)}$|§
    \begin§|figure|§
        \centering
        \includegraphics[width=0.95\linewidth]§|../{stat}/{filename}|§
        \vfill
        \includegraphics[width=0.95\linewidth]§|../{stat}/{filename.replace('.png', '_gain.png')}|§
    \end§|figure|§
\end§|frame|§""".replace("§|", "{").replace("|§", "}")

def stat_to_text(stat: str):
    _d = {
        "mi": "mutual information",
        "linearinfo": "mutual information under multivariate Gaussian assumption",
        "linearinfogauss": "mutual information under multivariate Gaussian assumption and after reparametrization"
    }
    return _d.get(stat) or stat

content = r"""
\documentclass{beamer}

\usepackage[english]{babel}
\usepackage[utf8x]{inputenc}

\mode<presentation>
{
  \usetheme{Rochester}
  \usecolortheme{default}
}

\AtBeginSection[]
{
  \begin{frame}{Outline}
    \tableofcontents[currentsection]
  \end{frame}
}

\usepackage{graphicx}

\setbeamersize{text margin left=5mm,text margin right=5mm} 

\title{""" + f"Informativity on ${plotter.params_comb_formatter(param)}$ ({stat_to_text(stat)})" + r"""}
\author{Lucas Einig}
\institute{IRAM - GIPSA-lab}
\date{\today}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}{Outline}
  \tableofcontents
\end{frame}
"""

for line in lines:
    files = [f for f in os.listdir(os.path.join(script_path, stat)) if f.endswith(".png") and not f.endswith("gain.png")]
    files = [f for f in files if line in f and f.startswith(param)]
    files.sort()

    ls = [f.replace(f"{param}__", "").replace(f"_{stat}.png", "").split("_") for f in files]

    content += "\n\n\section{Combinations including " + f"${plotter.lines_comb_formatter(line)}$ ({len(ls)} available)" + "}\n"
    content += "\n".join([
        build_frame(f, _ls) for f, _ls in zip(files, ls) if len(ls) > 1
    ])

content += "\n\end{document}"

presentation_name = f"{param}_{stat}_{date.today()}.tex"
with open(os.path.join(beamers_path, presentation_name), "wt") as f:
    f.write(content)

commands = [
    f"cd beamers && pdflatex {presentation_name} && pdflatex {presentation_name}",
    "cd beamers && rm -f *.aux *.log *.nav *.out *.snm *.toc"
]

for com in commands :
    os.system("/bin/bash -c \"" + com + "\"")
