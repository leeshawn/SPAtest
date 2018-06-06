\name{SPAmeta}
\alias{SPAmeta}
\title{
Meta-analysis using SPA test summary statistics
}
\description{
Performs meta-analysis using summary statistics obtained from SPA test within individual studies. It can be used to hybridize different meta-analysis techniques based on the available summary statistics.
}
\usage{
SPAmeta(pvalue.Z=NULL,MAF.Z=NULL,CCsize.Z=NULL,
	pvalue.GC=NULL,GCmat=NULL,CCsize.GC=NULL,Cutoff.GC=2,
	spldata=NULL,
	Cutoff.meta=2)
}
\arguments{
  \item{pvalue.Z}{
Vector denoting signed p values for which no other summary statistic is available and Z score conversion is suited.
}
  \item{MAF.Z}{
Vector denoting the minor allele frequencies in different studies corresponding to \code{pvalue.Z}. If missing, the different studies will be assumed to have no heterogeneity in the variants. Optional, if only Z score-based meta-analysis is being performed. Required, if being hybridized with other meta-analysis methods.
}
  \item{CCsize.Z}{
Matrix denoting the case-control sample sizes of the studies corresponding to \code{pvalue.Z}. Column 1 represents cases, column 2 controls.
}
  \item{pvalue.GC}{
Vector denoting signed p values for which genotype counts of the homozygous minor and heterozygous genotypes are available and the p values are calculated using the SPA test. These p values will be used in the meta-analysis using the genotype count-based meta-analysis strategy.
}
  \item{GCmat}{
Matrix denoting the genotype counts, column 1 represents the homozygous minor and column 2 the heterozygous genotypes.
}
  \item{CCsize.GC}{
Matrix denoting the case-control sample sizes of the studies corresponding to \code{pvalue.GC}. Column 1 represents cases, column 2 controls.
}
  \item{Cutoff.GC}{
An integer or vector specifying the Cutoffs used to calculate the SPA test p values within individual studies.
}
  \item{spldata}{
Vector, matrix or a list denoting the spline-based summary statistics as obtained from running \code{\link{ScoreTest_SPA_wMeta}} with \code{output} = "metaspline". Provide only for the studies with this information available.
}
  \item{Cutoff.meta}{
An integer denoting the fixed standard deviation cutoff to be used for the combined meta-analysis test. Default value is 2.
}
}

\details{
If all studies with genotype count information used the same cutoff, then providing only a single number as \code{Cutoff.GC} is sufficient. Otherwise, a vector denoting the cutoff within each study is required. Currently, "BE" as a within-study cutoff is not accepted.

If provided as a matrix, the rows of \code{spldata} should correspond to the individual studies for which spline-based meta-analysis is to be applied. Column 1 should denote the raw within-study scores, column 2 the corresponding variances. Out of the next 3k columns, first k columns should denote the nodes, next k columns the functional values of the first derivative of the CGF, and the last k columns the functional values of the second derivative of the CGF. None of the elements of the matrix can be missing or NA. If different number of nodes were used in different studies, please provide them in list format.
If spline-based summary statistic is available for only one study, then it can also be provided as a vector with 3k+2 elements where the elements denote similar statistics as denoted above in the case of a matrix.
If provided as a list, the elements of the list correspond to the different studies. Each element is a vector which denote the score, variance, nodes, and first and second derivative values in this order for that particular study. The list type input is more suited when the number of nodes in different studies are different.

If only Z score-based meta-analysis is needed, provide only \code{pvalue.Z}, \code{MAF.Z}, and \code{CCsize.Z}. \code{MAF.Z} is used to properly weight the Z scores when there is between study heterogeneity present. It is optional to provide the \code{MAF.Z} information if only Z score-based meta-analysis is being performed. Then the studies will be assumed to be homogeneous if \code{MAF.Z} is missing. However, \code{MAF.Z} is required if the Z score-based meta-analysis is being hybridized with other meta-analysis methods.
If only genotype count-based meta-analysis is needed, provide only \code{pvalue.GC},\code{GCmat},\code{CCsize.GC}, and \code{Cutoff.GC}.
If only spline-based meta-analysis is needed, provide only \code{spldata}.
If different studies provide different types of summary information, hybridize the methods accordingly.

For now, this package does not support having different minor alleles in different studies. If the minor alleles are different in different studies, some manual pre-processing is required, such as changing the signs of the p values, adjusting the MAF and genotype counts accordingly, and changing the signs of the scores, nodes and first derivatives in the spline-based summary statistics. In later implementations, we will try to provide a function for doing that pre-processing.
}

\value{
The signed meta-analysis p value is returned.
}
\author{
Rounak Dey, \email{deyrnk@umich.edu}
}
\seealso{
\code{\link{ScoreTest_wSaddleApprox_NULL_Model}}

\code{\link{ScoreTest_SPA_wMeta}}
}
\examples{
\dontrun{
SPAmeta(pvalue.Z=NULL,MAF.Z=NULL,CCsize.Z=NULL,
	pvalue.GC=NULL,GCmat=NULL,CCsize.GC=NULL,Cutoff.GC=2,
	spldata=NULL,
	Cutoff.meta=2)
}
}
\keyword{htest}
\keyword{regression}
\keyword{nonlinear}