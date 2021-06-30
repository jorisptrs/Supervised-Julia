
library(tidyverse)
library(ggplot2)

#setwd("~/Documents/Supervised-Julia/regression/cnn")

dat <- read.csv("predictions.csv", header=TRUE)
view(dat)

plot(dat$y_actual_real, dat$y_actual_img)

(p <- ggplot(dat, aes(x=y_actual_real, y=y_pred_real)) +
    geom_point() +
    ggtitle("") + 
    ylab("Predicted Re{c}") +
    xlab("Actual Re{c}") +
    geom_abline(intercept = 0, slope = 1, col="dodgerblue1")+
    scale_x_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    scale_y_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    theme(legend.position="none")
)

(p <- ggplot(dat, aes(x=y_actual_img, y=y_pred_img)) +
    geom_point() +
    ggtitle("") + 
    ylab("Predicted Im{c}") +
    xlab("Actual Im{c}") +
    geom_abline(intercept = 0, slope = 1, col="dodgerblue1")+
    scale_x_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    scale_y_continuous(limits=c(-1, 1),breaks = seq(-1, 1, by=0.25)) +
    theme(legend.position="none")
)
