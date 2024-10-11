# PhD - Machine Learning on Fish Oil Analysis

[![Documentation Status](https://readthedocs.org/projects/fishy-business/badge/?version=latest)](https://fishy-business.readthedocs.io/en/latest/?badge=latest)
[![Pylint](https://github.com/woodRock/fishy-business/actions/workflows/pylint.yml/badge.svg)](https://github.com/woodRock/fishy-business/actions/workflows/pylint.yml)

Something fishy is going on around here.

## Documentation

Read the documentation https://fishy-business.readthedocs.io/en/latest/?badge=latest

## Organisation

This repository is organized into the following directory structure:

```
.
├── code
├── docs
├── literature
├── papers
├── proposal
└── resources
```

(In an attempt) to organize my Phd thesis work (into a semblance of) order, we have folders:

- [**code**](code), which includes the documentated codebases used to produce the results of the experiments presented in this thesis.
- [**docs**](docs), documentation that organizes my literature reviews, minutes of meetings, thoughts, and miscellaneous, into an indexed/searchable and available anywhere anytime and on any device, website available online.
- [**literature**](literature), which contains a PDF filee for all the literature cited in my work. Ordered by their latex bib file name (e.g. wood2022automation), for sanity purposes.
- [**papers**](https://github.com/woodRock/fishy-business/tree/main/papers), the LaTeX source files for the papers I have written. 
- [**proposal**](proposal), the LaTeX document for my PhD proposal.
- [**resources**](resources), a dumping ground for anything that doesn't quite fit the literature category, but is nonetheless useful. E.g. presentation/lecture slides, article/magazine clippings.

## Playful

The existence, the physical universe is basically playful. There is no necessity for it whatsoever. It isn’t going anywhere. That is to say, it doesn’t have some destination that it ought to arrive at.

But it is best understood by analogy with music, because music, as an art form is essentially playful. We say, “You play the piano.” You don’t work the piano.

-- Alan Watts,"Coincidence of Opposites" in the Tao of Philosophy lecture series.

[![its-a-morray](https://user-images.githubusercontent.com/18411037/159612697-22525e7d-352d-444c-b746-5b94f5108449.jpeg)](https://www.youtube.com/watch?v=SezOrE0zRFo)


## Latex 

```latex
\begin{table}[H]
    \centering
    \caption{Classification results for species, part, oil and cross-species datasets}
    \rotatebox{90}{
        \begin{tabular}{l|l|l|l|l|l|l|l|l}
            \hline
            \multirow{2}{*}{Method} & \multicolumn{2}{c|}{Species} & \multicolumn{2}{c|}{Part} & \multicolumn{2}{c|}{Oil} & \multicolumn{2}{c}{Cross-species} \\
            & Train & Test & Train & Test & Train & Test & Train & Test \\
            \hline
            KNN & 
                95.76\% $\pm$ 0.00\% & 
                79.37\% $\pm$ 0.00\% & 
                43.06\% $\pm$ 0.00\% & 
                39.17\% $\pm$ 0.00\% & 
                55.30\% $\pm$ 0.00\% & 
                25.00\% $\pm$ 0.00\% & 
                81.50\% $\pm$ 0.00\% &
                59.38\% $\pm$ 0.00\% \\
            DT & 
                100.00\% $\pm$ 0.00\% &
                99.17\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% &
                35.50\% $\pm$ 4.35\% &
                100.00\% $\pm$ 0.00\% &
                23.40\% $\pm$ 2.06\% &
                100.00\% $\pm$ 0.00\% & 
                63.51\% $\pm$ 1.72\% \\
            LR  & 
                100.00\% $\pm$ 0.00\% &
                85.21\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% &
                59.58\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% &
                20.89\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% & 
                70.82\% $\pm$ 0.00\% \\
            LDA &
                98.54\% $\pm$ 0.00\% &
                92.29\% $\pm$ 0.00\% &
                74.31\% $\pm$ 0.00\% &
                52.92\% $\pm$ 0.00\% &
                70.07\% $\pm$ 0.00\% &
                22.86\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% &
                70.82\% $\pm$ 0.00\% \\
            NB  & 
                89.17\% $\pm$ 0.00\% &
                66.67\% $\pm$ 0.00\% & 
                100.00\% $\pm$ 0.00\% &
                48.33\% $\pm$ 0.00\% &
                65.48\% $\pm$ 0.00\% &
                25.54\% $\pm$ 0.00\% &
                72.12\% $\pm$ 0.00\% & 
                50.35\% $\pm$ 0.00\% \\
            RF  & 
                100.00\% $\pm$ 0.00\% &
                90.05\% $\pm$ 0.56\% &
                100.00\% $\pm$ 0.00\% &
                61.67\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% &
                32.45\% $\pm$ 2.32\% &
                100.00\% $\pm$ 0.00\% & 
                63.97\% $\pm$ 2.13\% \\ 
            SVM &
                100.00\% $\pm$ 0.00\% &
                84.58\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% &
                52.33\% $\pm$ 2.57\% &
                100.00\% $\pm$ 0.00\% &
                20.54\% $\pm$ 0.00\% &
                100.00\% $\pm$ 0.00\% & 
                69.51\% $\pm$ 0.00\% \\
            Ensemble & 
                100.00\% $\pm$ 0.00\% & 
                87.84\% $\pm$ 0.40\% &
                100.00\% $\pm$ 0.00\% &
                52.33\% $\pm$ 2.57\% &
                100.00\% $\pm$ 0.00\% &
                28.13\% $\pm$ 1.00\% &
                100.00\% $\pm$ 0.00\% & 
                68.76\% $\pm$ 1.24\% \\ 
            Transformer & & & & & & & & \\ 
            LSTM  & & & & & & & & \\
            VAE   & & & & & & & & \\
            KAN   & & & & & & & & \\
            CNN   & & & & & & & & \\
            Mamba & & & & & & & & \\
            MCIFC & & & & & & & & \\
            \hline
        \end{tabular}
    }
\end{table}
```