# M4R_public

\begin{itemize} 
\item Step 1: Initialise.
At time $t=2n$, define $X = x_{1}, \ldots, x_{n}$ with empirical distribution $P_{1}$, and $Y = y_{1}, \ldots, y_{n} = x_{n+1}, \ldots, x_{2n}$ with empirical distribution $P_{2}$, find

\begin{equation}
\arg \max _{A_{1} \in Gr(d, K)} M M D_{A}^{2}(P_{1}, P_{2})
\end{equation}

where
\begin{equation}
\operatorname{MMD}^{2}_{A_{1}}(P, Q) = \frac{1}{n^{2}} \sum_{i=1}^{n} \sum_{j=1}^{n} k\left(x_{i}A_{1}, x_{j}A_{1}\right)+\frac{1}{n^{2}} \sum_{i=1}^{n} \sum_{j=1}^{n} k\left(y_{i}A_{1}, y_{j}A_{1}\right)-\frac{2}{n^{2}} \sum_{i=1}^{n} \sum_{j=1}^{n} k\left(x_{i}A_{1}, y_{j}A_{1}\right)
\end{equation}

\item Step 2: At time $t=i\text{x}n$, $i$ and integer greater $2$, define $X = x_{t-2n}, \ldots, x_{t-n}$ with empirical distribution $P_{i-1}$, and $Y = y_{1}, \ldots, y_{n} = x_{t-n+1}, \ldots, x_{t}$ with empirical distribution $P_{i}$, find 

\begin{equation}
\arg \max _{A_{t} \in Gr(d, K)} M M D_{A_{t}}^{2}\left(P_{i-1}, P_{i}\right)-\lambda D\left(A_{t-1}, A_{t}\right),
\end{equation}

\item Step 3 Calculate the dimension reduced MMD using the optimal projection matrix $A_t^{opt}$, $M M D_{A_{t}^{opt}}^{2}(P_{i-1}, P_{i})$
\item Step 4 Repeat Step 2 and 3 as time progresses.
\end{itemize}
