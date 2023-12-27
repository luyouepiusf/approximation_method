library(optparse)
library(dplyr)
library(msm)

option_list<-list(
  make_option("--seedstart",type="character",default="1",metavar="character"),
  make_option("--seedend",type="character",default="100",metavar="character"))
opt_parser<-OptionParser(option_list=option_list)
opt<-parse_args(opt_parser)

seedstart<-as.numeric(opt$seedstart)
seedend<-as.numeric(opt$seedend)

nsim<-length(seedstart:seedend)

for(sim in 1:nsim){
  aseed<-(seedstart:seedend)[sim]
  print(aseed)
  
  ssij<-data.matrix(read.csv(paste0("data/ssij_",aseed,".csv")));ssij[ssij==-1]<-NA
  TTij<-data.matrix(read.csv(paste0("data/TTij_",aseed,".csv")));TTij[TTij==-1]<-NA
  zzi<-data.matrix(read.csv(paste0("data/zzi_",aseed,".csv")));colnames(zzi)<-paste0("z",1:ncol(zzi))
  
  nstates<-6
  zformula<-as.formula(paste0("~",paste(colnames(zzi),collapse="+")))
  
  df_state<-
    data.frame(id=c(row(ssij)),state=c(ssij),time=c(TTij)/100)%>%
    filter(!is.na(state))%>%
    arrange(id,time)
  df_covariate<-
    data.frame(id=1:nrow(zzi),zzi/2)
  df_join<-
    df_state%>%
    left_join(df_covariate,by="id")
  
  possible_transition<-
    rbind(
      c(F,T,T,F,F,F),
      c(F,F,F,T,T,F),
      c(F,F,F,F,T,T),
      c(F,F,F,F,F,F),
      c(F,F,F,F,F,F),
      c(F,F,F,F,F,F))
  Q<-possible_transition*0.01
  
  Q.crude<-crudeinits.msm(state~time,id,data=df_join,qmatrix=Q)
  
  fit_msm<-tryCatch(msm(
    state~time,
    subject=id,
    data=df_join,
    qmatrix=Q.crude,
    covariates=zformula),
    error=function(cond){
      message(paste0("Not converge",aseed))
      return(NULL)})
  if(!is.null(fit_msm)){
    write_beta<-data.frame()
    for(begin_idx in 1:nstates){
      for(end_idx in 1:nstates){
        if(!possible_transition[begin_idx,end_idx])next
        beta_est<-
          c(z1=coef(fit_msm)$z1[begin_idx,end_idx]/2,
            z2=coef(fit_msm)$z2[begin_idx,end_idx]/2,
            z3=coef(fit_msm)$z3[begin_idx,end_idx]/2,
            z4=coef(fit_msm)$z4[begin_idx,end_idx]/2)
        a_row_beta<-data.frame(seed=aseed,from=begin_idx,to=end_idx,t(beta_est))
        write_beta<-rbind(write_beta,a_row_beta)
      }
    }
    write.table(write_beta,"beta_multistate_msm.csv",append=T,col.names=F,row.names=F,sep=",")
  }
}





