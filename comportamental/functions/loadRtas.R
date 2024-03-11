loadRtas <- function(results){
  rtas = data.frame()
  mail = ""
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
  return(rtas)
}