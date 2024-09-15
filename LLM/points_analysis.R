library(ggplot2)
library(dplyr)
setwd('~/Desktop/Tesis/polisemia/LLM')
df = read.csv('processed_data.csv', sep =',')

ggplot(df) + 
  geom_point(aes(x=baseS1,y=SignS1, color = Model))+                           
  geom_point(aes(x=baseS2,y=SignS2, color = Model))+                          
  geom_abline(linetype="longdash") +                                            
  xlab("Sesgo Base") +                                                          
  ylab("Sesgo Generado") 

azules = df %>% filter(Model == 'GPT2 WordLevel') %>% select('Model', 'baseS1', 'SignS1', 'baseS2', 'SignS2',
                                                             "diff_base1_emb", "diff_base2_emb",
                                                              "indTarget",      "diff_base1" ,    "diff_base2")
rojos = df %>% filter(Model == 'GPT2 base') %>%  select('Model', 'baseS1', 'SignS1', 'baseS2', 'SignS2',
                                                             "diff_base1_emb", "diff_base2_emb",
                                                              "indTarget",      "diff_base1" ,    "diff_base2")

azules_arriba = sum(azules[,'baseS2'] < azules[,'SignS2']) + sum(azules[,'baseS2'] < azules[,'SignS2'])
rojos_arriba = sum(rojos[,'baseS2'] < rojos[,'SignS2']) + sum(rojos[,'baseS2'] < rojos[,'SignS2'])
print('Cantidad de datos azules arriba de la diagonal')
print(azules_arriba)
print('Cantidad de datos rojos arriba de la diagonal')
print(rojos_arriba)

ggplot(df) +                                                               
  geom_point(aes(x=diff_base1, y=diff_base1_emb, color = Model))+                           
  geom_point(aes(x=diff_base2, y=diff_base2_emb, color = Model))+                          
 geom_abline(linetype="longdash") +
  geom_vline(xintercept = 0)+
  geom_hline(yintercept = 0)+
  xlab("Sesgo humano (%)") +                                                          
  ylab("Sesgo embedding (%)")  
 
azules_arriba_emb = sum(azules[,'diff_base1'] < azules[,'diff_base1_emb']) + sum(azules[,'diff_base2'] < azules[,'diff_base2_emb'])
rojos_arriba_emb = sum(rojos[,'diff_base1'] < rojos[,'diff_base1_emb']) + sum(rojos[,'diff_base2'] < rojos[,'diff_base2_emb'])
print('Cantidad de datos azules arriba de la diagonal')
print(azules_arriba_emb)
print('Cantidad de datos rojos arriba de la diagonal')
print(rojos_arriba_emb)
