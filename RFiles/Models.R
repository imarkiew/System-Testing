require(frbs)

FRBCS.CHI <- function(num_of_labels, train, X, Xrange)
{
  method.type <- "FRBCS.CHI"
  control.FRBCS.CHI <- list(num.labels = num_of_labels, type.mf = "GAUSSIAN")
  object.FRBCS.CHI <- frbs.learn(train, Xrange, method.type, control.FRBCS.CHI)
  pred <- predict(object.FRBCS.CHI, X)
  pred <- round(pred, digits = 0)
  return(pred)
}

FRBCS.W <- function(num_of_labels, train, X, Xrange)
{
  method.type <- "FRBCS.W"
  control.FRBCS.W <- list(type.mf = "GAUSSIAN", num.class = num_of_labels)
  object.FRBCS.W <- frbs.learn(train, Xrange, method.type, control.FRBCS.W)
  pred <- predict(object.FRBCS.W, X)
  pred <- round(pred, digits = 0)
  return(pred)
}




	