#Read data
graph <- read.csv("Graph.csv")
seeds <- read.csv("Seed.csv",header=F)
data <- read.csv("Extracted_features.csv",header=F)
#data <- read.csv("XPCA.csv",header=F)

##Kmeans

#Try PCA first

pcares <- prcomp(data)

png("PCA_elbow.png")
plot(1:1084,pcares$sdev,xlab="Dimension",ylab="Eigenvalues")
abline(v=30,lty=2)
dev.off()

#Choose first 30 PCs

points <- pcares$x[6001:10000,1:30]
points2 <- pcares$x[1:6000,1:30]
centroids <- pcares$x[seeds[,1],1:30]

#Kmeans

km <- kmeans(points,centroids,iter.max=100,algorithm="Lloyd")

#Merge 60 clusters to 10
result <- c()
for (c in km$cluster) {
  result <- c(result,seeds[c,2])
}

#Plot clustering results
png("pca_clusters.png")
plot(points[,1],points[,2],col=result,xlab="Eigenvector 1",ylab="Eigenvector 2")
dev.off()

pred <- cbind(6001:10000,result)
colnames(pred) <- c("Id","Label")

write.csv(pred,file="prediction4.csv",row.names=F)

#Spectral Clustering Code

#Make A
adjMat <- matrix(rep(0,6000*6000),ncol=6000)

for (i in 1:dim(graph)[1]) {
  n1 <- graph[i,1]
  n2 <- graph[i,2]
  adjMat[n1,n2] <- 1
  adjMat[n2,n1] <- 1
}

#save(adjMat,file="adjMat.rda")

#Make D
d <- matrix(rep(0,6000*6000),ncol=6000)
for (i in 1:6000) {
  d[i,i] <- sum(adjMat[i,])
}

#Make D^-1/2

"%^%" <- function(M, power)
  with(eigen(M), vectors %*% (values^power * solve(vectors)))

d_test <- d %^% (-1/2)

d12 <- matrix(rep(0,6000*6000),ncol=6000)
for (i in 1:6000) {
  d12[i,i] <- 1 / sqrt(d[i,i])
}

#Make Lap mat

l <- diag(6000) - (d_test %*% adjMat %*% d_test)

save(l,file="lapMat.rda")

#Get eigenvectors

e <- eigen(l)

#save(e,file="eigenvectors2.rda")

#Make elbow plot for spectral clustering
png("Spectral_elbow.png")
plot(1:6000,e$values,xlab="Dimensions",ylab="Eigenvalues")
abline(v=5971,lty=2)
dev.off()

#Cluster with k means

specCentroids <- e$vectors[seeds[,1],5971:6000]
kmresult <- kmeans(e$vectors[,5971:6000],specCentroids,iter.max=1000,algorithm="Lloyd")

result <- c()
for (c in kmresult$cluster) {
  result <- c(result,seeds[c,2])
}

#Plot result
png("spec_cluster_plot.png")
plot(e$vectors[,6000],e$vectors[,5999],col=result,xlab="Eigenvector 1",ylab="Eigenvector 2")
dev.off()

##Try m clust
library(mclust)

mod1 <- MclustDA(points2,result)
mod2 <- MclustDA(points2,result,modelType="EDDA")

predict_result <- predict(mod2,points)

pred <- cbind(6001:10000,as.numeric(as.character(predict_result$classification)))
colnames(pred) <- c("Id","Label")

write.csv(pred,file="prediction6.csv",row.names=F)

#Try weakly supervised retraining method
#You must do this part by hand carefully, it's not automated since we have to check results are each retrain step

newLabels <- predict(mod2,points2)$classification
mod3 <- MclustDA(points2,as.numeric(as.character(newLabels)),modelType="EDDA")

predict_result <- predict(mod3,points)$classification

pred <- cbind(6001:10000,as.numeric(as.character(predict_result)))
colnames(pred) <- c("Id","Label")

write.csv(pred,file="prediction12.csv",row.names=F)

retrainModel <- function(mod) {
  newLabs <- predict(mod,points2)$classification
  newMod <- MclustDA(points2,as.numeric(as.character(newLabs)),modelType="EDDA")
  return(newMod)
}

mod4 <- retrainModel(mod3)
predict_result <- predict(mod4,points)$classification

mod5 <- retrainModel(mod4)
predict_result <- predict(mod5,points)$classification

mod6 <- retrainModel(mod5)
predict_result <- predict(mod9,points)$classification


##Try cca

embeddings <- e$vectors[,5971:6000]

X1 <- e$vectors[,5971:6000]
X2 <- pcares$x[1:6000,1:30]
cca <- function(mat, d){
  covar <- cov(mat)
  v1v1 <- covar[1:(dim(covar)[1]/2),1:(dim(covar)[2]/2)]
  v1v2 <- covar[1:(dim(covar)[1]/2),(dim(covar)[2]/2 + 1):dim(covar)[2]]
  v2v1 <- covar[(dim(covar)[1]/2 + 1):dim(covar)[1],1:(dim(covar)[2]/2)]
  v2v2 <- covar[(dim(covar)[1]/2 + 1):dim(covar)[1],(dim(covar)[2]/2 + 1):dim(covar)[2]]
  
  w <- ginv(v1v1) %*% v1v2 %*% ginv(v2v2) %*% v2v1
  v <- ginv(v2v2) %*% v2v1 %*% ginv(v1v1) %*% v1v2
  
  eigw <- eigen(w)
  eigv <- eigen(v)
  
  x1 <- mat[,1:(dim(mat)[2]/2)]
  x2 <- mat[,(dim(mat)[2]/2 + 1):dim(mat)[2]]
  
  x1 <- sweep(x1,2,colMeans(x1))
  x2 <- sweep(x1,2,colMeans(x2))
  
  we <-  x1 %*% eigw$vectors[,1:d]
  ve <-  x2 %*% eigv$vectors[,1:d]
  
  return (list(v1=we,v1=ve))
}

cca_result <- cca(cbind(X1,X2),2)

#View 1
plot(cca_result[[1]],rep(0,6000),col=km_result_30dim$cluster)

#View 2
plot(cca_result[[2]],rep(0,6000),col=km_result_30dim$cluster)

