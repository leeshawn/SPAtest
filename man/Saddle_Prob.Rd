\name{Saddle_Prob}
\alias{Saddle_Prob}
\alias{Saddle_Prob_fast}
\title{
Calculate Saddlepoint p values (for external libraries)
}
\description{
Function to calculate the SPA p value from score function. Intended to be used by external libraries.
}
\usage{
Saddle_Prob(q, mu, g, Cutoff=2,alpha,output="P",nodes.fixed,nodes.init,log.p=FALSE)
Saddle_Prob_fast(q, g,mu,gNA,gNB,muNA,muNB,Cutoff=2,alpha,output,nodes.fixed,
nodes.init,log.p=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{q}{
Numeric scalar, score statistic.
}
  \item{mu}{
Numeric vector, fitted probabilities from the null model.
}
  \item{g}{
Numeric vector, covariate adjusted genotypes.
}
  \item{gNA}{
Numeric vector, covariate adjusted genotypes where the observed genotype is zero.
}
  \item{gNB}{
Numeric vector, covariate adjusted genotypes where the observed genotype is non-zero.
}
  \item{muNA}{
Numeric vector, fitted probabilities from the null model where the observed genotype is zero.
}
  \item{muNB}{
Numeric vector, fitted probabilities from the null model where the observed genotype is non-zero.
}
  \item{Cutoff}{
An integer or the string "BE" denoting the standard deviation cutoff to be used. If \code{Cutoff} = "BE", the level-specific cutoff based on Berry-Esseen theorem is calculated. If the test statistic lies within the standard deviation cutoff of the mean, p-value based on traditional score test is returned. Default value is 2.
}
  \item{alpha}{
Significance level for the test(s), default value is \eqn{5\times 10^{-8}}. Used only if \code{Cutoff} = "BE".
}
  \item{output}{
String specifying the output required. Possible values are "\code{P}" (default), "\code{metaZ}", "\code{metaGC}", and "\code{metaspline}".
}
  \item{nodes.fixed}{
Vector denoting the spline nodes for the spline based summary statistics, if you do not want to provide a fixed set of nodes instead of estimating the optimal set of nodes. Only applicable when the output is "\code{metaspline}".
}
  \item{nodes.init}{
Vector denoting the initial values of the spline nodes when you want to estimate the optimal set of spline nodes using the coordinate descent algorithm. Only applicable when the output is "\code{metaspline}". Ignored if  \code{nodes.fixed} is provided. The node at 0 will be automatically added, no need to provide that. The number of finally selected nodes will be the same as in \code{nodes.init}.
}
  \item{log.p}{
Whether to return natural log-transformed p-values, default value is \code{FALSE}.
}
}
%\details{
%}
\value{
\item{p.value}{p-value or natural log-transformed p-value based on the saddlepoint approximation. If \code{output} = "P", it is unsigned. For all other choice of \code{output}, it is signed.}
\item{p.value.NA}{p-value or natural log-transformed p-value based on the normal approximation (traditional score test).  If \code{output} = "P", it is unsigned. For all other choice of \code{output}, it is signed.}
\item{Is.converge}{"TRUE" or "FALSE" denoting whether the root-finding algorithm for the saddlepoint equation has converged.}
\item{Score}{Centered score statistic.}
\item{splfun}{Outputs from fitting the CGF-Spline method, if \code{output} = "metaspline".}
\item{var}{Variance of the score test statistic, if \code{output} = "metaspline".}
}

\seealso{
\code{\link{ScoreTest_SPA}}

\code{\link{ScoreTest_SPA_wMeta}}
}
\examples{
\dontrun{
Saddle_Prob(q, mu, g, Cutoff=2,alpha,output="P",nodes.fixed,nodes.init,log.p=FALSE)
Saddle_Prob_fast(q, g,mu,gNA,gNB,muNA,muNB,Cutoff=2,alpha,output,nodes.fixed,nodes.init,log.p=FALSE)
}
}
\keyword{regression}
\keyword{models}