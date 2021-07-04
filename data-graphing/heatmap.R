
library('plot.matrix')

d <- read.csv("risks.csv", header=TRUE)

A = matrix(log(d$risk), nrow=4,ncol=5,byrow = TRUE)  
rownames(A) <-  c(0.0001, 0.001, 0.01, 0.1)
colnames(A) <- c(0, 0.0001, 0.001, 0.01, 0.1)
par(mai=c(1, 1, 0.8, 1))
plot(A, digits=3, text.cell=list(cex=1),xlab="Weight Decay", 
     ylab="Learning Rate", main="Learning Rate and Weight Decay on Model Performance")
legend("topright", legend="Log risk",bty = "n")
