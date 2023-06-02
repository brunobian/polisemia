read.pcibex <- function(filepath, auto.colnames=TRUE, fun.col=function(col,cols){cols[cols==col]<-paste(col,"Ibex",sep=".");return(cols)}) {
  n.cols <- max(count.fields(filepath,sep=",",quote=NULL),na.rm=TRUE)
  if (auto.colnames){
    cols <- c()
    con <- file(filepath, "r")
    while ( TRUE ) {
      line <- readLines(con, n = 1, warn=FALSE)
      if ( length(line) == 0) {
        break
      }
      m <- regmatches(line,regexec("^# (\\d+)\\. (.+)\\.$",line))[[1]]
      if (length(m) == 3) {
        index <- as.numeric(m[2])
        value <- m[3]
        if (is.function(fun.col)){
          cols <- fun.col(value,cols)
        }
        cols[index] <- value
        if (index == n.cols){
          break
        }
      }
    }
    close(con)
    return(read.csv(filepath, comment.char="#", header=FALSE, col.names=cols))
  }
  else{
    return(read.csv(filepath, comment.char="#", header=FALSE, col.names=seq(1:n.cols)))
  }
}
library("dplyr")
library("ggplot2")
library("maditr")


setwd("/data/brunobian/Documents/Repos/Repos_Analisis/polisemia/")
results <- read.pcibex("results_1_648.csv")
results$Value <- as.character(results$Value)
results$PennElementName <- as.character(results$PennElementName)

rtas = data.frame()
for (i in 1:dim(results)[1]){
  thisLine = results[i,]
  if (thisLine[,"PennElementName"] == "mail" & thisLine[,"Value"] != ""){
    mail =  results[i,"Value"]
  }
  thisLine$mail = mail
  if (results[i,"PennElementName"] == "opciones"){
    rtas =  rbind(rtas,thisLine)
  }
}


# rtas = results[results$PennElementName=="opciones",]

rtas$IDOrac <- as.numeric(as.character(rtas$IDOrac))
rtas$indTarget = rtas$IDOrac %% 100
rtas$contextType = rtas$IDOrac %/% 100

rtas$acc <- (rtas$Value == rtas$SignEsperado) | (rtas$Value == paste0(rtas$SignEsperado,2))
rtas <- rtas[,c("IDOrac","SignEsperado","Target","acc","indTarget","contextType","mail")]

# Check acc suj: miro sujs que tienen baja acc en Contexto 1 y 2 (los sesgadores)
porSuj <- rtas[rtas$contextType<3,] %>%  group_by(mail) %>% 
            summarise( n = n(), 
                       mean = mean(acc)
            )
filtSujAcc = porSuj[porSuj[porSuj$n>0,]$mean < .2,]$mail

# Check catch suj: miro sujs que hicieron catch trials (tipo 4)
porSuj <- rtas[rtas$contextType==4,] %>%  group_by(mail) %>% 
  summarise( n = n())
filtSujCatch = setdiff(unique(rtas$mail),porSuj$mail)

filtSujs = unique(c(filtSujCatch,filtSujAcc))

rtas$filtSuj <- is.element(rtas$mail, filtSujs)

rtasFilt <- rtas[rtas$filtSuj==FALSE,]

###

agrupadas <- rtasFilt %>%  group_by(IDOrac) %>% 
             summarise(mean = mean(acc), 
                       n = n(),
                       indTarget = mean(indTarget),
                       contextType = mean(contextType))

plot(agrupadas$IDOrac,agrupadas$mean)
plot(agrupadas$IDOrac,agrupadas$n)

porContext <- agrupadas[agrupadas$contextType < 3,]

ggplot(porContext, aes(x=indTarget,y=mean,col=contextType)) +
  geom_point(size=2, shape=23)


verDiff <-  data.frame(dcast(porContext, 
                      formula = indTarget~contextType,
                      fun.aggregate = sum,value.var = "mean"))

verDiff["diff"] <- abs( verDiff["X1"] - verDiff["X2"])

umbralDiff = .6
umbralAcc = .3
verDiff$umbral =  verDiff["diff"] > umbralDiff |
                  verDiff["X1"]   < umbralAcc | 
                  verDiff["X2"]   < umbralAcc

ggplot(verDiff, aes(x=indTarget,y=diff,col=umbral)) +
  geom_point(size=2, shape=23) + 
  geom_hline(yintercept = umbral)

verDiff[verDiff["umbral"]==TRUE,]

