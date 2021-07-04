
library(tidyverse)
library(ggplot2)

dat <- read.csv("predictions.csv", header=TRUE)
view(dat)

plot(dat$y_actual_real, dat$y_actual_img)

(p_real <- ggplot(dat, aes(x=y_actual_real, y=y_pred_real)) +
    geom_point() +
    ggtitle("Comparison of Real Part") + 
    ylab("Predicted Re{c}") +
    xlab("Actual Re{c}") +
    geom_abline(intercept = 0, slope = 1, col="dodgerblue1", size=1, alpha=0.8)+
    scale_x_continuous(limits=c(-2, 2),breaks = seq(-2, 2, by=0.5)) +
    scale_y_continuous(limits=c(-2, 2),breaks = seq(-2, 2, by=0.5)) +
    theme(legend.position="none", plot.title = element_text(hjust = 0.5))
)

ggsave("comp_real.pdf", plot=p_real, width=7, height=5, units="in")

(p_img <- ggplot(dat, aes(x=y_actual_img, y=y_pred_img)) +
    geom_point() +
    ggtitle("Comparison of Imaginary Part") + 
    ylab("Predicted Im{c}") +
    xlab("Actual Im{c}") +
    geom_abline(intercept = 0, slope = 1, col="dodgerblue1", size=1, alpha=0.8)+
    scale_x_continuous(limits=c(-2, 2),breaks = seq(-2, 2, by=0.5)) +
    scale_y_continuous(limits=c(-2, 2),breaks = seq(-2, 2, by=0.5)) +
    theme(legend.position="none", plot.title = element_text(hjust = 0.5))
)

ggsave("comp_img.pdf", plot=p_img, width=7, height=5, units="in")

mod1 <- lm(data=dat, y_pred_img~y_actual_img)
summary(mod1)

mod2 <- lm(data=dat, y_pred_real~y_actual_real)
summary(mod2)

