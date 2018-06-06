\name{ScoreTest_SPA_wMeta}
\alias{ScoreTest_SPA_wMeta}
\title{
Prepare summary statistics for meta-analysis and perform SPA test.
}
\description{
Performs score test using saddlepoint approximation and prepares summary statistics for meta-analysis.
}
\usage{
ScoreTest_SPA_wMeta(genos,pheno,cov,obj.null,method=c("fastSPA","SPA"),minmac=5,
Cutoff=2,alpha=5*10^-8,missing.id=NA,beta.out=FALSE,beta.Cutoff=5*10^-7,
output=c("P","metaZ","metaGC","metaspline"),nodes.fixed=NULL,
nodes.init=c(-100,-10,-1,1,10,100))
}
\arguments{
  \item{genos}{
A vector or matrix containing the genotypes or dosages. If matrix is provided then rows should correspond to SNPs and columns should correspond to subjects. Optional, but needed if \code{obj.null} is missing.
}
  \item{pheno}{
A vector containing the outcomes (phenotypes). Optional, but needed if \code{obj.null} is missing.
}
  \item{cov}{
A matrix or data frame containing the covariates. Optional, but needed if \code{obj.null} is missing.
}
  \item{obj.null}{
An object of class "\code{SA_NULL}". (Optional)
}
  \item{method}{
String specifying the p-value calculation method. Possible values are "\code{fastSPA}" (default) and "\code{SPA}".
}
  \item{minmac}{
Minimum minor allele count threshold to run SPA test, default value is \eqn{5}.
}
  \item{Cutoff}{
An integer or the string "BE" denoting the standard deviation cutoff to be used. If \code{Cutoff} = "BE", the level-specific cutoff based on Berry-Esseen theorem is calculated. If the test statistic lies within the standard deviation cutoff of the mean, p-value based on traditional score test is returned. Default value is 2.
}
  \item{alpha}{
Significance level for the test(s), default value is \eqn{5\times 10^{-8}}. Used only if \code{Cutoff} = "BE".
}
  \item{missing.id}{
Missing value indicator. Numeric or \code{NA}, default value is \code{NA}.
}
  \item{beta.out}{
Logical indicating whether log odds ratios (beta parameters) are to be estimated, default value is \code{FALSE}.
}
  \item{beta.Cutoff}{
Maximum p-value threshold for beta parameters to be estimated, default value is \eqn{5\times 10^{-7}}.
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
}

\details{
\code{genos} can have discrete \code{0, 1, 2} values or continuous values between \code{[0,2]}. The genotype or dosage values can represent any of the major allele, minor allele, reference allele or alternate allele counts (or dosages), as long as it is consistent throughout the subjects.

\code{genos} can have missing values denoted by the \code{missing.id} argument. Such missing values will be imputed using mean imputation. \code{pheno} or \code{cov} cannot have missing values. 

\code{pheno} and \code{cov} are ignored if \code{obj.null} is provided. If both \code{obj.null} and \code{cov} is missing, or \code{obj.null} is missing and \code{cov=NULL}, then the vector \code{rep(1,n)} is assigned to \code{cov}, where \code{n} is the number of subjects.

\code{method} = "SPA" is the basic saddlepoint approximation based test without the partially normal approximation improvement.
\code{method} = "fastSPA" utilizes the partially normal approximation approach for improved efficiency, especially for rare variants.

Beta parameters are estimated using Firth's method, and thus computationally expensive. Therefore, it is recommended that beta parameters are only to be estimated when the p-value is very small (denoted by \code{beta.Cutoff}). The code for beta estimation is as implemented by Clement Ma in the EPACTS software.

If \code{output} = "P", the output will be the same as running \code{\link{ScoreTest_SPA}}. No summary statistics for meta-analysis is provided.

If \code{output} = "metaZ", the output will be the same as with \code{output} = "P", except the p values in the output will be signed, and minor allele frequencies are also provided. This choice of \code{output} provide required summary statistics for Z score-based meta-analysis.

If \code{output} = "metaGC", the output will be the same as with \code{output} = "P", except the p values in the output will be signed, and genotype counts of homozygous minor and heterozygous genotypes are also provided.  This choice of \code{output} provide required summary statistics for genotype count-based meta-analysis.

If \code{output} = "metaspline", the output will be the same as with \code{output} = "metaGC", additionally spline-based summary statistics are also provided.  This choice of \code{output} provide required summary statistics for spline-based as well as genotype count-based meta-analysis.
}

\value{
\item{p.value}{p-value based on the saddlepoint approximation. If \code{output} = "P", it is unsigned. For all other choice of \code{output}, it is signed.}
\item{p.value.NA}{p-value based on the normal approximation (traditional score test).  If \code{output} = "P", it is unsigned. For all other choice of \code{output}, it is signed.}
\item{Is.converge}{"TRUE" or "FALSE" denoting whether the root-finding algorithm for the saddlepoint equation has converged.}
\item{beta}{Genotype log-odds ratio estimate.}
\item{SEbeta}{Standard error for the genotype log-odds ratio.}
\item{MAF}{Minor allele frequencies. Only provided when \code{output} = "metaZ".}
\item{GCmat}{Genotype counts for homozygous minor (column 1) and heterozygous genotypes (column 2). Only provided when \code{output} = "metaGC" or "metaspline".}
\item{spldata}{Spline-based summary statistics on the CGF. Column 1 represents the raw score values, column 2 the variance of those scores, and the next columns represent nodes, first and second derivatives of the CGF in equal sizes. Only provided when \code{output} = "metaspline".}
}
\author{
Rounak Dey, \email{deyrnk@umich.edu}
}
\references{
  Dey, R. et al., 2017. A Fast and Accurate Algorithm to Test for Binary Phenotypes and Its Application to PheWAS.
  The American Journal of Human Genetics, Vol 101 (1), 37-49.
  
  Ma, C. et al., 2013. Recommended Joint and Meta-Analysis Strategies for Case-Control Association Testing of Single Low-Count Variants.
  Genetic Epidemiology, Vol 37 (6), 539-550.
}
\seealso{
\code{\link{ScoreTest_wSaddleApprox_NULL_Model}}

\code{\link{ScoreTest_SPA}}

\code{\link{SPAmeta}}
}
\examples{
\dontrun{
ScoreTest_SPA_wMeta(genos,pheno,cov,obj.null,method=c("fastSPA","SPA"),
minmac=5,Cutoff=2,alpha=5*10^-8,missing.id=NA,beta.out=FALSE,beta.Cutoff=5*10^-7,
output=c("P","metaZ","metaGC","metaspline"),nodes.fixed=NULL,
nodes.init=c(-100,-10,-1,1,10,100))
}
}
\keyword{htest}
\keyword{regression}
\keyword{nonlinear}