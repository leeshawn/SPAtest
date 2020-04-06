\name{ScoreTest_SPA}
\alias{ScoreTest_SPA}
\title{
Score test based on saddlepoint approximation
}
\description{
Performs score test using saddlepoint approximation to estimate the null distribution.
}
\usage{
ScoreTest_SPA(genos,pheno,cov,obj.null,method=c("fastSPA","SPA"),minmac=5,
Cutoff=2,alpha=5*10^-8,missing.id=NA,beta.out=FALSE,beta.Cutoff=5*10^-7,log.p=FALSE)
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
  \item{log.p}{
Whether to return natural log-transformed p-values, default value is \code{FALSE}. NOTE: log-transformed p-values are not compatible with Meta-analysis using this package.
}
}

\details{
\code{genos} can have discrete \code{0, 1, 2} values or continuous values between \code{[0,2]}. The genotype or dosage values can represent any of the major allele, minor allele, reference allele or alternate allele counts (or dosages), as long as it is consistent throughout the subjects.

\code{genos} can have missing values denoted by the \code{missing.id} argument. Such missing values will be imputed using mean imputation. \code{pheno} or \code{cov} cannot have missing values. 

\code{pheno} and \code{cov} are ignored if \code{obj.null} is provided. If both \code{obj.null} and \code{cov} is missing, or \code{obj.null} is missing and \code{cov=NULL}, then the vector \code{rep(1,n)} is assigned to \code{cov}, where \code{n} is the number of subjects.

\code{method} = "SPA" is the basic saddlepoint approximation based test without the partially normal approximation improvement.
\code{method} = "fastSPA" utilizes the partially normal approximation approach for improved efficiency, especially for rare variants.

Beta parameters are estimated using Firth's method, and thus computationally expensive. Therefore, it is recommended that beta parameters are only to be estimated when the p-value is very small (denoted by \code{beta.Cutoff}). The code for beta estimation is as implemented by Clement Ma in the EPACTS software.
}

\value{
\item{p.value}{p-value or natural log-transformed p-value based on the saddlepoint approximation.}
\item{p.value.NA}{p-value or natural log-transformed p-value based on the normal approximation (traditional score test).}
\item{Is.converge}{"TRUE" or "FALSE" denoting whether the root-finding algorithm for the saddlepoint equation has converged.}
\item{beta}{Genotype log-odds ratio estimate.}
\item{SEbeta}{Standard error for the genotype log-odds ratio.}
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

\code{\link{ScoreTest_SPA_wMeta}}
}
\examples{
\dontrun{
ScoreTest_SPA(genos,pheno,cov,obj.null,method=c("fastSPA","SPA"),
minmac=5,Cutoff=2,alpha=5*10^-8,missing.id=NA,beta.out=FALSE,beta.Cutoff=5*10^-7,log.p=FALSE)
}
}
\keyword{htest}
\keyword{regression}
\keyword{nonlinear}