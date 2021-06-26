
library(tidyverse)
library(ggplot2)

dat <- read.csv("../nn/viktor/predictions.csv", header=TRUE)
view(dat)

(p <- ggplot(dat, aes(x=y_true_real, y=y_pred_real)) +
    geom_point() +
    ggtitle("") + 
    ylab("Predicted Re{c}") +
    xlab("Actual Re{c}") +
    geom_abline(intercept = 0, slope = 1, col="dodgerblue1")+
    scale_x_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    scale_y_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    theme(legend.position="none")
)

(p <- ggplot(dat, aes(x=y_true_img, y=y_pred_img)) +
    geom_point() +
    ggtitle("") + 
    ylab("Predicted Im{c}") +
    xlab("Actual Im{c}") +
    geom_abline(intercept = 0, slope = 1, col="dodgerblue1")+
    scale_x_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    scale_y_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    theme(legend.position="none")
)
