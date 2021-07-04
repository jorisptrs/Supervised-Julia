
df <- read.csv("../trainingData/labels.txt", header=FALSE)

pdf(file = "./julia_random.pdf",   # The directory you want to save the file in
    width = 10, # The width of the plot in inches
    height = 10)
par(cex.lab=1.5, cex.main=1.6)
plot(df$V1, df$V2, xlab="Real", main="Sampled Julia constants", ylab="Imaginary")
dev.off()

plot(density(df$V1))

plot(density(df$V2))

ks.test(df$V1,"punif",-1,1)

l =  read.csv("../regression/cnn/loss_final.csv", header=TRUE)
n = length(l$val_loss)
pdf(file = "./loss_final.pdf",   # The directory you want to save the file in
    width = 10, # The width of the plot in inches
    height = 10)
par(cex.lab=1.6, cex.main=1.4)
plot(
  1:n,
  l$train_loss,
  type="l",
  col="blue",
  log="x",
  xlab="Log epochs",
  ylab="Loss",
  main="Final training loss (800 epochs)",
  lwd=2.5
)
lines(1:n, l$val_loss, col="#f29100", lwd=4.5)
legend(x="topright", inset=0.04, legend=c("Training loss", "Validaiton loss"),
       col=c("blue", "#f29100"), lty=1:1, cex=1.2,
       box.lty=1, lwd=4.5)
dev.off()

require(ggplot2)
require(plotly)

l = read.csv("../regression/testData/loss.csv", header=TRUE)

print(min(l$loss))
print(max(l$loss))
print(mean(l$loss))
n = 400 


ggplot(l, aes(x=yreal, y=yimag, group=loss, color=loss)) + geom_area() + geom_point()

plot(l$yreal, l$yimag, xlab="Real", main="Sampled Julia constants", ylab="Imaginary")

ggplot(l, aes(x=wt, y=mpg, color=cyl)) + geom_point()

length(l$yreal)

require(tidyverse)
df <- read.csv("./data/risks.csv", header=TRUE)
df <- as_tibble(df)

for (i in 1:20) {
  if(i %% 2 == 0) {
    next
  } 
  
  df <- df %>% filter(combination!=i)
}
df

require(plotly)

len <- length(df$risk)

scene = list(camera = list(eye = list(x = -1.25, y = 1.25, z = 1.25)))

fig <- plot_ly(
  x=df$learning.rate,
  y=df$alpha,
  z=df$risk,
  intensity=df$risk,
  type = 'mesh3d', 
  colors = colorRamp(c("blue", "red"))
)
fig <- layout(fig, 
              xaxis = list(type = "linear"),
              yaxis = list(type = "linear"),
              scene = scene      
)
fig

plot_ly(
  x=df$learning.rate,
  y=df$alpha,
  z=predict(lo),
  type = 'scatter3d', 
  colors = df$risk
)

