require(caTools)
require(unbalanced)
require(DMwR)
require(fmsb)
require(stats)
require(clusterSim)
source("Models.R")

prepare_data <- function(is_category_numerical, delimiter, path)
{
  Xx <- read.table(paste(path, "/DividedSets/", "Xx", sep=""), sep = delimiter, head = FALSE)
  yy <- read.table(paste(path, "/DividedSets/", "yy", sep=""), sep = delimiter, head = FALSE)
  Xt <- read.table(paste(path, "/DividedSets/", "Xt", sep=""), sep = delimiter, head = FALSE)
  yt <- read.table(paste(path, "/DividedSets/", "yt", sep=""), sep = delimiter, head = FALSE)
  if(!is_category_numerical)
  {
    yy[, 1] <- unclass(yy[, 1])
    yt[, 1] <- unclass(yt[, 1])
  }
  is_zero_present = FALSE
  for(i in 1:NROW(yy))
  {
    if(yy[i, 1] == 0)
    {
      is_zero_present = TRUE
      break
    }
  }
  for(i in 1:NROW(yt))
  {
    if(yt[i, 1] == 0)
    {
      is_zero_present = TRUE
      break
    }
  }
  if(is_zero_present)
  {
    for(i in 1:NROW(yy))
    {
      yy[i, 1] <- yy[i, 1] + 1
    }
    for(i in 1:NROW(yt))
    {
      yt[i, 1] <- yt[i, 1] + 1
    }
  }
  Xy <-Xx
  Xy$class <- yy
  Xrange <- sapply(Xx, range)
  write.table(yt, file = paste(path, "/RResults/yt", sep=""), append=FALSE, col.names=FALSE, row.names = FALSE)
  return(list(matrix(as.numeric(unlist(Xt)), nr=nrow(Xt)), matrix(as.numeric(unlist(Xy)), nr=nrow(Xy)), Xrange))
}

  run_tests <- function(is_category_numerical, delimiter, num_of_labels, path)
{
  data <-prepare_data(is_category_numerical, delimiter, path)
  X <- data[[1]]
  train <- data[[2]]
  Xrange <-data[[3]]
  pred_chi <- FRBCS.CHI(num_of_labels, train, X, Xrange)
  pred_w <- FRBCS.W(num_of_labels, train, X, Xrange)
  write.table(pred_chi, file = paste(path, "/RResults/pred_chi", sep=""), append=FALSE, col.names=FALSE, row.names = FALSE)
  write.table(pred_w, file = paste(path, "/RResults/pred_w", sep=""), append=FALSE, col.names=FALSE, row.names = FALSE)
}