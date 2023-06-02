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

setwd("/home/bbianchi/Downloads/PCIbex/")
results <- read.pcibex("results_1_648.csv")

rtas = results[results$PennElementName=="opciones",]

rtas$acc <- (rtas$Value == rtas$SignEsperado) | (rtas$Value == paste0(rtas$SignEsperado,2))
rtas <- rtas[,c("IDOrac","SignEsperado","Target","acc")]

group_by(rtas,"IDOrac")

agrupadas <- rtas %>%  group_by(IDOrac) %>% 
             summarise(mean = mean(acc), n = n())

plot(agrupadas$IDOrac,agrupadas$mean)
plot(agrupadas$IDOrac,agrupadas$n)
