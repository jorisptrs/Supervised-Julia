


dat500 <- read.csv("loss500.csv", header = TRUE)
dat1000 <- read.csv("loss1000.csv", header = TRUE)
dat5000 <- read.csv("loss5000.csv", header = TRUE)
dat10000 <- read.csv("loss10000.csv", header = TRUE)

epochs  =  seq(1, 25, by = 1)

colors <- c("500"  =  "black", "1000"  =  "red", "5000"  =  "blue", "10000"  =  "green4")

(p <- ggplot() + 
    geom_line(aes(epochs, dat500$train_loss), color="red") + 
    annotate("text", label="N=500", x=6.7, y=0.075, color="red", size=3) +
    geom_line(aes(epochs, dat500$val_loss), linetype  =  "dashed", color="red") +
    geom_line(aes(epochs, dat1000$train_loss), color="blue") + 
    annotate("text", label="N=1000", x=3.8, y=0.068, color="blue", size=3) +
    geom_line(aes(epochs, dat1000$val_loss), linetype  =  "dashed", color="blue") +
    geom_line(aes(epochs, dat5000$train_loss), color="green4") + 
    annotate("text", label="N=5000", x=2.8, y=0.03, color="green4", size=3) +
    geom_line(aes(epochs, dat5000$val_loss), linetype  =  "dashed", color="green4") +
    geom_line(aes(epochs, dat10000$train_loss), color="black") + 
    annotate("text", label="N=10000", x=1.3, y=0.005, color="black", size=3) +
    geom_line(aes(epochs, dat10000$val_loss, linetype  =  "Validation"),color="black") +
    xlab("Epoch") +
    ylab("Loss") +
    ggtitle("Dataset Size (N) on Model Performance") +
    scale_y_continuous(breaks  =  seq(0, 0.2, by = 0.05)) +
    scale_x_continuous(breaks  =  seq(1, 25, by = 2)) +
    scale_linetype_manual(values  =  c("Training"="solid", "Validation"="dashed")) +
    theme(legend.position="bottom",legend.title = element_blank(), plot.title = element_text(hjust = 0.5))
  )

ggsave("dataset_size.pdf", plot=p, width=7, height=5, units="in")
