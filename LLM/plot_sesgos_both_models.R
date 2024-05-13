library(ggplot2)
setwd("/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/")
df=read.csv("distancias_nuevo_modelo.csv",sep=",")

df[,"baseS1"] <- as.numeric(as.character(df[,"X2"]))
df[,"SignS1"] <- as.numeric(as.character(df[,"X3"]))
df[,"baseS2"] <- as.numeric(as.character(df[,"X5"]))
df[,"SignS2"] <- as.numeric(as.character(df[,"X6"]))

ggplot(df) +                                                               
  geom_point(aes(x=baseS1,y=SignS1), col="#00CDCD")+                           
  geom_point(aes(x=baseS2,y=SignS2), col= "#00CDCD")+                          
  geom_abline(linetype="longdash") +                                            
  xlab("Sesgo Base") +                                                          
  ylab("Sesgo Generado")

ggsave("base_generado.png", plot = p)
df["diff_base1_emb"] <- (df["SignS1"] - df["baseS1"]) / abs(df["baseS1"])
df["diff_base2_emb"] <- (df["SignS2"] - df["baseS2"]) / abs(df["baseS2"])

df2 <- read.csv2("../comportamental/accuracies.csv",sep=",")
df2 <- df2[,c("indTarget","diff_base1","diff_base2")]

df2[,"diff_base1"] <- as.numeric(as.character(df2[,"diff_base1"]))
df2[,"diff_base2"] <- as.numeric(as.character(df2[,"diff_base2"]))

df <- merge(df,df2,by.x="X0",by.y="indTarget")

ggplot(df) +                                                               
  geom_point(aes(x=diff_base1,y=diff_base1_emb), col="#00CDCD")+                           
  geom_point(aes(x=diff_base2,y=diff_base2_emb), col= "#00CDCD")+                          
  geom_abline(linetype="longdash") +
  geom_vline(xintercept = 0)+
  geom_hline(yintercept = 0)+
  xlab("Sesgo humano (%)") +                                                          
  ylab("Sesgo embedding (%)") + 
  xlim(c(-0.1,0.5))+
  ylim(c(-1,2))

ggsave("humano_embedding.png", plot = q)
