args = commandArgs(trailingOnly=TRUE)
path = args[1]
setwd(paste(path, "/RFiles", sep=""))
source("Tools.R")

delimiter <- ","
if(args[2] == "true")
{
    is_category_numerical = TRUE
}else
{
    is_category_numerical = FALSE
}
num_of_labels <- as.numeric(args[3])

run_tests(is_category_numerical, delimiter, num_of_labels, path)
