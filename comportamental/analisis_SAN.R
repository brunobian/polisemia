rm(list = ls())
setwd("/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/comportamental")
source("functions/read.pcibex.R")
source("functions/loadRtas.R")
library("dplyr")
library("ggplot2")
library("maditr")
library("reshape2")


# Load results from Ibex ####
results       <- read.pcibex("data/results_final.csv")
results$Value <- as.character(results$Value)
results$PennElementName <- as.character(results$PennElementName)

# Generate responses, with additional info ####
rtas <- loadRtas(results)
# rtasOrig <- rtas
# rtasOrig -> rtas
rtas$SignEsperado <- as.character(rtas$SignEsperado)
rtas$IDOrac      <- as.numeric(as.character(rtas$IDOrac))
rtas$indTarget   <- rtas$IDOrac %% 100
rtas$contextType <- rtas$IDOrac %/% 100

# Extract both "meanings" for each target word (indTarget)
significados <-  data.frame(dcast(rtas[rtas$contextType<4,], 
                                  formula = indTarget~contextType,
                                  fun.aggregate = function(x) x[1], value.var = "SignEsperado"))
significados <- significados[,-4]
significados <- significados %>% rename("Sign1" = "X1","Sign2" = "X2")

rtas = merge(rtas, significados, by="indTarget", all=TRUE)

# Calculate accuracies (ibex renamed some options with a 2 at the end)
rtas$acc   <- (rtas$Value == rtas$SignEsperado) | (rtas$Value == paste0(rtas$SignEsperado,2))
rtas$otro  <- (rtas$Value == "Otro")            | (rtas$Value == "Otro2")
rtas$accS1 <- (rtas$Value == rtas$Sign1)        | (rtas$Value == paste0(rtas$Sign1,2))
rtas$accS2 <- (rtas$Value == rtas$Sign2)        | (rtas$Value == paste0(rtas$Sign2,2))

# keep only relevant columns
rtas <- rtas[,c("IDOrac","SignEsperado","Target","acc",
                "indTarget","contextType","mail",
                "Oracion","otro","accS1","accS2")]
write.csv(rtas,"data/rtas.csv",row.names=FALSE)

# [FILTERS] ####
rtas <- read.csv("data/rtas.csv")

# Check acc suj: remove subjects with low acc in biasing Contexts 1 y 2 
porSuj <- rtas[rtas$contextType<3,] %>%  group_by(mail) %>% 
  summarise( n = n(), 
             mean = mean(acc)
  )
filtSujAcc = porSuj[porSuj[porSuj$n>0,]$mean < .2,]$mail

# Check catch suj: remove subjects without catch trials (tipo 4) 
porSuj <- rtas[rtas$contextType==4,] %>%  group_by(mail) %>% 
  summarise( n = n())
filtSujCatch = setdiff(unique(rtas$mail),porSuj$mail)

filtSujs = unique(c(filtSujCatch,filtSujAcc))

rtas$filtSuj <- is.element(rtas$mail, filtSujs)

rtasFilt <- rtas[rtas$filtSuj==FALSE,]
write.csv(rtasFilt,"data/rtasFilt.csv",row.names=FALSE)

# Analyses ####
rtasFilt <- read.csv("data/rtasFilt.csv")

agrupadas <- rtasFilt %>%  group_by(IDOrac) %>% 
  summarise(mean = mean(acc), 
            sd   = sd(acc)/sqrt(n()), 
            otro = mean(otro),
            n    = n(),
            indTarget   = mean(indTarget),
            contextType = mean(contextType),
            oracion     = first(Oracion),
            m_baseS1    = mean(accS1),
            m_baseS2    = mean(accS2),
            sd_baseS1   = sd(accS1)/sqrt(n()),
            sd_baseS2   = sd(accS2)/sqrt(n())
  )

# plot(agrupadas$IDOrac,agrupadas$mean)
# plot(agrupadas$IDOrac,agrupadas$otro)
# plot(agrupadas$IDOrac,agrupadas$n)
agrupadas$contextTypeStr <- agrupadas$contextType
agrupadas$contextTypeStr[agrupadas$contextType==1] = "Contexto1"
agrupadas$contextTypeStr[agrupadas$contextType==2] = "Contexto2"
agrupadas$contextTypeStr[agrupadas$contextType==3] = "Neutro"
agrupadas$contextTypeStr[agrupadas$contextType==4] = "CatchTrial"

ggplot(agrupadas, aes(x=IDOrac, y=mean, color=contextTypeStr )) + 
  geom_point()+
  geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.1,
                position=position_dodge(0.05))

porContext <- agrupadas[agrupadas$contextType < 3,c("IDOrac","mean","otro","n","indTarget","contextType","contextTypeStr","oracion")]

ggplot(porContext, aes(x=indTarget,y=mean,col=contextTypeStr)) +
  geom_point(size=2, shape=23)

# Extraigo info de acc para Sig1 y Sig2
verDiff <-  data.frame(dcast(porContext, 
                             formula = indTarget+oracion~contextType,
                             fun.aggregate = list(sum,sum) , value.var = list("mean")))
verDiff <- verDiff %>% 
  rename("Sign1" = "X1",
         "Sign2" = "X2")

# Extraigo info de acc base para Sig1 y Sig2 de las respuestas de Context3
neutras <- agrupadas[agrupadas$contextType == 3,]
neutras <- merge(neutras, significados, by="indTarget", all=TRUE)
verDiff <- merge(verDiff, neutras[,c("indTarget","m_baseS1","m_baseS2")], by="indTarget", all=TRUE)

head(verDiff)

verDiff["diff"]       <- abs( verDiff["Sign1"] - verDiff["Sign2"])
verDiff["diff_base1"] <- (verDiff["Sign1"] - verDiff["m_baseS1"]) /verDiff["m_baseS1"]
verDiff["diff_base2"] <- (verDiff["Sign2"] - verDiff["m_baseS2"]) /verDiff["m_baseS2"]

ggplot(verDiff) +
  geom_point(aes(x=m_baseS1,y=Sign1), col="#00CDCD")+
  geom_point(aes(x=m_baseS2,y=Sign2), col= "#00CDCD")+
  geom_abline(linetype="longdash") +
  xlab("Sesgo Base") + 
  ylab("Sesgo Generado")

# LLMs ####
setwd("/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/LLM/")
df=read.csv2("distancias.csv",sep=",")

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

