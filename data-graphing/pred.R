
library(tidyverse)

dat <- read.csv("../nn/viktor/predictions.csv", header=TRUE)
view(dat)

plot(dat$y_true_real, dat$y_pred_real, main="real part", ylab="predicted real value", xlab="true real value")
plot(dat$y_true_img, dat$y_pred_img, main="imaginary part", ylab="predicted imaginary value", xlab="true imaginary value")
