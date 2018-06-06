\name{ScoreTest_wSaddleApprox_NULL_Model}
\alias{ScoreTest_wSaddleApprox_NULL_Model}
\title{
Preparing the null model
}
\description{
Prepares the null model object \code{SA_NULL} to be used in \code{ScoreTest_SPA} based on the outcome and the covariates.
}
\usage{
ScoreTest_wSaddleApprox_NULL_Model(formula, data=NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{
An object of class "formula" (as used in the function "glm").
}
  \item{data}{
An optional data frame, list or environment containing the variables in the model.
}
}
%\details{
%}
\value{
\code{ScoreTest_wSaddleApprox_NULL_Model} returns an object of class \code{SA_NULL}.
}
\author{
Rounak Dey, \email{deyrnk@umich.edu}
}

\seealso{
\code{\link{ScoreTest_SPA}}

\code{\link{ScoreTest_SPA_wMeta}}
}
\examples{
\dontrun{
ScoreTest_wSaddleApprox_NULL_Model(formula, data=NULL)
}
}
\keyword{regression}
\keyword{models}