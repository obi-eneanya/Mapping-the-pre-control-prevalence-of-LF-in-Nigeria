# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~ Modelling LF prevalence in Nigeria using a machine learning approach ~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
## Date: 07.2018
## Place: London, UK
## Project: Modelling LF prevalence in Nigeria using Quantile Regression Forest

## Section 1: Set up a function to install and load multiple R packages.
# Check to see if packages are installed. Install them if they are not, then load them into the R session.

ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# List of packages
packages <- c("sp","raster","dismo","maptools","rgdal","proj4","ggplot2","tidyr","usdm",
              "readxl","caret","psych","randomForest","miscTools","quantregForest",
              "PrevMap","ggmap","mapview","tmap","sf")

ipak(packages)

# 1.1 - Import shapefiles
shapefiles <- setwd("C:/Users/oae11/Desktop/NG_LF_ENM/Shapefiles")
Africa_ADM0 <- readOGR(dsn=shapefiles, layer="SALB_master_L0")
Africa_ADM1 <- readOGR(dsn=shapefiles, layer="SALB_master_L1")

Nigeria.ADM0 <- subset(Africa_ADM0, Africa_ADM0$ADM0_NAME == "Nigeria")
Nigeria.ADM1 <- subset(Africa_ADM1, Africa_ADM1$ADM0_NAME == "Nigeria")
plot(Nigeria.ADM0)
plot(Nigeria.ADM1)

# 1.2 - Load dataset and plot survye locations over shapefile
# Read in complete LF dataset which includes ICT and Mf prevalences 

ICT_NG <- read.csv("LF_NG_Complete.csv")
names(ICT_NG)

# Covert Diagnostic.Type to Factor
ICT_NG$Diagnostic.Type <- as.factor(ICT_NG$Diagnostic.Type)

summary(ICT_NG$Prevalence)
hist(ICT_NG$Prevalence)

points(ICT_NG$Longitude, ICT_NG$Latitude, pch=21, cex = 0.8, col="red") 
points(Mf_NG$Longitude, Mf_NG$Latitude, pch=21, cex = 0.8, col="red") 

## 1.3 - Apply empirical logit transformation on prevalence data: spread the distribution to get
## a quasi continuous distribution

ICT_NG$LogPrev <- log((ICT_NG$Positive + 0.5)/(ICT_NG$Examined - ICT_NG$Positive + 0.5))

summary(ICT_NG$LogPrev)
hist(ICT_NG$LogPrev)

# 1.4 - Clean dataset. 
# Remove NAs and surveys conducted out of the expected period of time
# in order to account for any potential temporal trend

i <- which(ICT_NG$Year_start == 1900 | is.na(ICT_NG$Year_start))
ICT_NG <- ICT_NG[-i,]

table(ICT_NG$Year_start, useNA = "ifany")


# Section 2: Tuning up a model based on Quantile Regression Forest for ICT prevalence

# 2.1 Run some basic random forest model to start exploring options for this modelling approach
# Remove potential missing values in predictors

names(ICT_NG)
i <- complete.cases(ICT_NG[, c(1,2, 19, 22:39)])
table(i) # not missing data

ICT_NG <- na.omit(ICT_NG[, c(1,2, 19, 22:39)])
names(ICT_NG)

# 2.2 - Variogram plot to assess correlation of prevalence points

ICT_coords <- as.matrix(ICT_NG[, c("Longitude", "Latitude")])
ICT_variogram <- variog(coords = ICT_coords, data = ICT_NG$LogPrev,
                        uvec = c(0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2))
plot(ICT_variogram)

ICT_Emp_variogram <- variofit(ICT_variogram, ini.cov.pars = c(2, 0.2), 
                              cov.model ="matern", fix.nugget = FALSE, 
                              nugget = 0, fix.kappa = TRUE, kappa = 0.3)
lines(ICT_Emp_variogram)

# 2.3 - Run simple randomForest model

set.seed(300)
ICT.RF.v00 <- randomForest(formula = LogPrev ~ ., 
                           data = ICT_NG[,4:22],
                           importance = T)
plot(ICT.RF.v00)

# Number of trees: 500
# No. of variables tried to each split: 9
# Mean of squared residuals: 1.174763
# % Var explained: 51.5

varImpPlot(ICT.RF.v00, type = 1)


## 2.4 - Tune up parameters for the random Forest model

ctrl <- trainControl(method = "repeatedcv",
                     number = 10, repeats = 5) # 10-fold CV repeated 5 times

grid_ref <- expand.grid(.mtry = c(3,5,7,9,12,15,18))

set.seed(300)
system.time(m_rf <- train(LogPrev ~ ., 
                          data = ICT_NG[4:22], 
                          method = "rf",
                          metric = "RMSE", 
                          trControl = ctrl,
                          tuneGrid = grid_ref))


# user  system elapsed (16.5 min)
# 990.35    7.80  998.99 

# Random Forest 

# 1103 samples
# 17 predictor

# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times) 
# Summary of sample sizes: 993, 994, 993, 991, 992, 993, ... 
# Resampling results across tuning parameters:

# mtry  RMSE      Rsquared   MAE      
# 3    1.200980  0.4142982  0.9369524
# 5    1.196854  0.4167875  0.9304566***
# 7    1.197480  0.4153966  0.9285400
# 9    1.199202  0.4132903  0.9296433
# 12   1.200143  0.4119367  0.9285512
# 15   1.202211  0.4098565  0.9299522
# 17   1.201592  0.4103760  0.9283960

# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 5.

# Run a random Forest model based on optimized mtry = 5

set.seed(300)
ICT.RF.v02 <- randomForest(LogPrev ~ ., 
                           data = ICT_NG[4:22],
                           mtry = 5,
                           importance = T)

# Number of trees: 500
# No. of variables tried to each split: 5
# Mean of squared residuals: 1.389594
# % Var explained: 42.96

plot(ICT.RF.v02)
varImpPlot(ICT.RF.v02, type = 1)


# Section 3 -  Use Quantile Regression Forest for final modelling of prevalence 

# Fit a model using a train dataset based on 75% of data and predict over a
# a heldout sample for a test dataset

# 3.1 - Divide into training and test data: ICT data

names(ICT_NG)

n <- nrow(ICT_NG)
set.seed(123)
indextrain <- sample(1:n,round(0.75*n),replace=FALSE)

ICT_train <- ICT_NG[indextrain,]
ICT_test <- ICT_NG[-indextrain,]


# 3.2 Prepare the X (features or predictors) and Y (response) for train and test datasets

X.ICT.train <- ICT_train[,4:21]
Y.ICT.train <- ICT_train[,22]

X.ICT.test <- ICT_test[,4:21]
Y.ICT.test <- ICT_test[,22]

str(X.ICT.train)
str(X.ICT.test)


# 3.3 - Train a model for ICT data
X.ICT.train <- X.ICT.train[, -ncol(X.ICT.train)]
set.seed(300)
QRF.NG.ICT.v01 <- quantregForest(x = X.ICT.train, y = Y.ICT.train, 
                                 mtry = 5,
                                 importance = T)


# 3.4 - Display the variable importance based on the %IncMSE

varImpPlot(QRF.NG.ICT.v01, 
           type = 1, 
           sort = T,
           main = "Variable Importance for Quantile Regression Forest")

Imp.Cov <- as.data.frame(importance(QRF.NG.ICT.v01, type = 1))
Imp.Cov <- Imp.Cov[order(Imp.Cov$`%IncMSE`),, drop = FALSE]


feat.names <- c("Monthly coldest temperature", "Flow accumulation", "Monthly warmest temperature", "Wetness index", 
                "Distance to permanent rivers", "Night light emissivity", "Terrain slope", "Soil pH", "Distance to stable     lights", "Clay soil content", "Elevation", "Enhanced vegetation index", "Silt soil content", "Land surface temperature", "Wettest quarter precipitation", "Distance to permanent water bodies",
"Driest quarter precipitation", "Diagnostic type")

row.names(Imp.Cov) <- feat.names

dotchart(Imp.Cov$`%IncMSE`,
         labels=rownames(Imp.Cov),
         pt.cex= 1.5,
         pch = 19,
         main="Variable Importance for quantile regression forest model", 
         xlab="% Increment MSE by variable permutation")

setwd(path_outputs)

dev.print(file="Variable_Importance_ICT.png", device=png, width=900)
dev.off()

setwd(path_wd)

# 3.5 - Implement cross-validation using the heldout subsample (test dataset)

condit.mean.v01 <- predict(QRF.NG.ICT.v01, X.ICT.test, 
                           what = mean, 
                           all=T)

# 3.6 - Estimating the R-square (as the fraction of the total sum of squares that is explained
# by the trained random forest)

SSE.v01 <- sum((Y.ICT.test - condit.mean.v01)^2)
TSS.v01 <- sum((Y.ICT.test - mean(Y.ICT.test))^2)

rsq.v01 <- 1-(SSE.v01/TSS.v01) # 0.4049531
RMSE.v01 <- sqrt(SSE.v01/length(Y.ICT.test)) # 1.236427

rm(SSE.v01, TSS.v01)

# 3.7 - Run some validation of the final prediction: extract residuals and run some analysis
# Normality test on the residuals

ICT.RF.residuals <- QRF.NG.ICT.v01$y - QRF.NG.ICT.v01$predicted

ICT.RF.Outputs <- as.data.frame(cbind(Observed = QRF.NG.ICT.v01$y, 
                                      Predicted = QRF.NG.ICT.v01$predicted, 
                                      Residuals = ICT.RF.residuals))

hist(ICT.RF.Outputs$Residuals)
boxplot(ICT.RF.Outputs$Residuals)

plot(X.ICT.train$Longitude, ICT.RF.Outputs$Residuals)   # To plot this. add Longitude in X.ICT.train above
abline(lm(ICT.RF.Outputs$Residuals ~ X.ICT.train$Longitude), col = "red")   # To plot this. add Longitude in X.ICT.train above


# 3.8 - Explore for spatial autocorrelation on the residuals.
# Function below is for plotting semivariogram for the residuals

ggvario <- function(coords, data, bins = 15, color = "royalblue1") {
  empvario <- variog(coords = coords, data = data, uvec = seq(0, max(dist(coords))/3, l = bins), messages = F)
  envmc <- variog.mc.env(geodata = as.geodata(cbind(coords, data)), obj.variog = empvario, nsim = 99, messages = F)
  dfvario <- data.frame(distance = empvario$u, empirical = empvario$v,
                        lowemp = envmc$v.lower, upemp = envmc$v.upper)
  ggplot(dfvario, aes(x = distance)) +
    geom_ribbon(aes(ymin = lowemp, ymax = upemp), fill = color, alpha = .3) +
    geom_point(aes(y = empirical), col = "black", fill = color, shape = 21, size = 3) +
    xlab("distance") +
    scale_y_continuous(name = "semivariance", breaks = seq(0, max(dfvario$upemp), .5), 
                       limits = c(0, max(dfvario$upemp))) +
    ggtitle("Empirical semivariogram") +
    theme_classic()  
}

source("ggvario.R")

# Plot semivariogram demonstrating presence/absence of autocorrelation on the residuals. 
ggvario(coords = X.ICT.train[,c(1,2)], #To plot variogram add Longitude in X.ICT.train above
        data = ICT.RF.Outputs$Residuals)
        

# Section 4 - Using QRF to produce continuous surface of predicted median and credible intervals for ICT prev 

# 4.1 - Predict over continuous surface of predictors and obtain the credible intervals and predicted median

# 4.1.1 - For ICT
set.seed(123)
system.time(Predict.ICT.v01 <- predict(QRF.NG.ICT.v01, newdata = ICT_pred_cov,
                                       what = c(0.025, 0.5, 0.975)))

set.seed(123)
system.time(Predict.ICT.mean <- predict(QRF.NG.ICT.v01, newdata = ICT_pred_cov,
                                        what = mean))

## 4.1.2 - Generate the predicted continuous surface for LB, median and UB

Predict.ICT.output <- as.data.frame(cbind(ICT_pred_cov[,c(1,2)], Predict.ICT.v01, Predict.ICT.mean))
colnames(Predict.ICT.output) <- c("x","y","LB","Median","UB","Mean")
names(Predict.ICT.output)

# 4.1.3 - Tranform back the predicted values (divide by 100 to estimate cases by 100,000 inhab)

Predict.ICT.output[,3:6] <- (psych::logistic(Predict.ICT.output[,3:6]))*100

# 4.1.4 - Generate raster for LB

Predicted.ICT.LB <- Predict.ICT.output[,c(1,2,3)]
coordinates(Predicted.ICT.LB) <- ~ x + y # create spatial points data frame
proj4string(Predicted.ICT.LB) <- proj4string(raster.map)
gridded(Predicted.ICT.LB) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.ICT.LB <- raster(Predicted.ICT.LB) # coerce to raster

plot(Predicted.ICT.LB)

# 4.1.5 - Generate raster for median

Predicted.ICT.Median <- Predict.ICT.output[,c(1,2,4)]
coordinates(Predicted.ICT.Median) <- ~ x + y # create spatial points data frame
proj4string(Predicted.ICT.Median) <- proj4string(raster.map)
gridded(Predicted.ICT.Median) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.ICT.Median <- raster(Predicted.ICT.Median) # coerce to raster

plot(Predicted.ICT.Median)

# 4.1.6 - Generate rater for UB

Predicted.ICT.UB <- Predict.ICT.output[,c(1,2,5)]
coordinates(Predicted.ICT.UB) <- ~ x + y # create spatial points data frame
proj4string(Predicted.ICT.UB) <- proj4string(raster.map)
gridded(Predicted.ICT.UB) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.ICT.UB <- raster(Predicted.ICT.UB) # coerce to raster

plot(Predicted.ICT.UB)

# 4.1.7 - Generate rater for mean

Predicted.ICT.Mean <- Predict.ICT.output[,c(1,2,6)]
coordinates(Predicted.ICT.Mean) <- ~ x + y # create spatial points data frame
proj4string(Predicted.ICT.Mean) <- proj4string(raster.map)
gridded(Predicted.ICT.Mean) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.ICT.Mean <- raster(Predicted.ICT.Mean) # coerce to raster

plot(Predicted.ICT.Mean)

#to give all predicted rasters same CRS
crs(Predicted.ICT.LB) <- crs(Predicted.ICT.Median)



# 4.1.8 - Pack all predicted rasters in a stack object

Prediction.ICT <- stack(Predicted.ICT.LB, Predicted.ICT.Median, 
                        Predicted.ICT.UB, Predicted.ICT.Mean)

crs(Nigeria.ADM0) <- crs(Prediction.ICT)

Prediction.ICT <- mask(Prediction.ICT, Nigeria.ADM0)

plot(Prediction.ICT)

Prediction.ICT <- projectRaster(Prediction.ICT, crs = PCS)

plot(Prediction.ICT)

## 4.1.9 - Export raster datasets as outputs

setwd(path_outputs)
dir.create("ICT_models")
setwd("./ICT_models")

writeRaster(Prediction.ICT, format = "GTiff", bylayer = T, names(Prediction.ICT), overwrite = T)

## Save workspace with objects created
gc()
save.image(paste0(path_wd, sep = "/","Nigeria_LF_modelling.RData"))


# 4.2 - For Mf

set.seed(123)
system.time(Predict.Mf.v01 <- predict(QRF.NG.Mf.v01, newdata = Mf_pred_cov,  #note that 'pred_cov' might be different as it was 
                                      what = c(0.025, 0.5, 0.975)))       #re-generated during bartMachine implementation  

set.seed(123)
system.time(Predict.Mf.mean <- predict(QRF.NG.Mf.v01, newdata = Mf_pred_cov,
                                       what = mean))

## 4.2.1 - Generate the predicted continuous surface for LB, median and UB

Predict.Mf.output <- as.data.frame(cbind(ICT_pred_cov[,c(1,2)], Predict.Mf.v01, Predict.Mf.mean))
colnames(Predict.Mf.output) <- c("x","y","LB","Median","UB","Mean")
names(Predict.Mf.output)

# 4.2.2 - Tranform back the predicted values (divide by 100 to estimate prevalence rate)

Predict.Mf.output[,3:6] <- (psych::logistic(Predict.Mf.output[,3:6]))*100

# 4.2.3 - Generate raster for LB

Predicted.Mf.LB <- Predict.Mf.output[,c(1,2,3)]
coordinates(Predicted.Mf.LB) <- ~ x + y # create spatial points data frame
proj4string(Predicted.Mf.LB) <- proj4string(raster.map)
gridded(Predicted.Mf.LB) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.Mf.LB <- raster(Predicted.Mf.LB) # coerce to raster

plot(Predicted.Mf.LB)

# 4.2.4 - Generate raster for median

Predicted.Mf.Median <- Predict.Mf.output[,c(1,2,4)]
coordinates(Predicted.Mf.Median) <- ~ x + y # create spatial points data frame
proj4string(Predicted.Mf.Median) <- proj4string(raster.map)
gridded(Predicted.Mf.Median) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.Mf.Median <- raster(Predicted.Mf.Median) # coerce to raster

plot(Predicted.Mf.Median)

# 4.2.5 - Generate rater for UP

Predicted.Mf.UB <- Predict.Mf.output[,c(1,2,5)]
coordinates(Predicted.Mf.UB) <- ~ x + y # create spatial points data frame
proj4string(Predicted.Mf.UB) <- proj4string(raster.map)
gridded(Predicted.Mf.UB) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.Mf.UB <- raster(Predicted.Mf.UB) # coerce to raster

plot(Predicted.Mf.UB)

# 4.3.6 - Generate rater for mean

Predicted.Mf.Mean <- Predict.Mf.output[,c(1,2,6)]
coordinates(Predicted.Mf.Mean) <- ~ x + y # create spatial points data frame
proj4string(Predicted.Mf.Mean) <- proj4string(raster.map)
gridded(Predicted.Mf.Mean) <- TRUE # coerce to SpatialPixelsDataFrame
Predicted.Mf.Mean <- raster(Predicted.Mf.Mean) # coerce to raster

plot(Predicted.Mf.Mean)

#to give all predicted rasters same CRS
#crs(Predicted.Mf.LB) <- crs(Predicted.Mf.Median)

# 4.2.7 - Pack all predicted rasters in a stack object

Prediction.Mf <- stack(Predicted.Mf.LB, Predicted.Mf.Median, 
                       Predicted.Mf.UB, Predicted.Mf.Mean)

crs(Nigeria.ADM0) <- crs(Prediction.Mf)

Prediction.Mf <- mask(Prediction.Mf, Nigeria.ADM0)

plot(Prediction.Mf)

Prediction.Mf <- projectRaster(Prediction.Mf, crs = PCS)

## 4.2.8 - Export raster datasets as outputs

setwd(path_outputs)
dir.create("Mf_models")
setwd("./Mf_models")

writeRaster(Prediction.Mf, format = "GTiff", bylayer = T, names(Prediction.Mf), overwrite = T)

## Save workspace with objects created
gc()
save.image(paste0(path_wd, sep = "/","Nigeria_LF_modelling.RData"))


# Section 5 - Finding correlations between ICT Obs and Predicted

# 5.1 Rasterise the ICT_NG and Mf_NG dataframes, retaining only Long/Lat, Prevalence and LogPrev colums

# For ICT ~ this preojects ICT and Mf data as spDataframe

ICT.cor <- ICT_NG[,1:3]
LF.sp.ICT <- SpatialPoints(ICT.cor[,c('Longitude', 'Latitude')], proj4string = CRS("+proj=longlat +datum=WGS84"))
ICT.cor <- SpatialPointsDataFrame(LF.sp, ICT.cor)
ICT.cor <- spTransform(ICT.cor, PCS)
str(ICT.cor)
class(ICT.cor)
head(ICT.cor[,1:3])

Mf.cor <- Mf_NG[,1:3]
LF.sp.Mf <- SpatialPoints(Mf.cor[,c('Longitude', 'Latitude')], proj4string = CRS("+proj=longlat +datum=WGS84"))
Mf.cor <- SpatialPointsDataFrame(LF.sp.Mf, Mf.cor)
Mf.cor <- spTransform(Mf.cor, PCS)
str(Mf.cor)
class(Mf.cor)
head(Mf.cor[,1:3])


# 5.2 - Extract Observed prevalence values with the corresponding predicted values. 

Obs.ICT.Pred.ICT <- raster::extract(Prediction.ICT$Mean, ICT.cor) #extracting Obs ICT<->Pred ICT

Obs.Mf.Pred.Mf <- raster::extract(Prediction.Mf$Mean, Mf.cor) #extracting Obs Mf<->Obs Mf


# 5.3 - Correlation tests 

cor.test(ICT.cor$Prevalence, Obs.ICT.Pred.ICT) #correlation between Obs ICT and Pred ICT

cor.test(Mf.cor$Prevalence, Obs.Mf.Pred.Mf) #correlation between Obs Mf and Pred Mf


# Section 6 - Function for plotting graphs of observed vs predicted values and 95% prediction intervals

ggvalidate <- function(observed, predicted, CI, type, alpha = 1) {
  # Checks
  if(ncol(CI) != 2) stop("CI must be a matrix or a data.frame with two columns")
  # Build the data frame with the values to plot
  df <- data.frame(observed, predicted, CI)
  names(df)[3:4] <- c("low", "up")
  df$outside <- 1 - as.numeric(with(df, observed >= low & observed <= up))
  
  if(type == 1) {
    # Plot observed vs. predicted with confidence intervals
    ggplot(df, aes(y = predicted, x = observed, col = factor(outside))) +
      geom_abline(intercept = 0, slope = 1, linetype = 2) +
      geom_pointrange(aes(ymin = low, ymax = up), alpha = alpha, size = .2) +
      scale_color_manual("", label = c("Inside", "Outside"),
                         values = ggsci::pal_lancet("lanonc")(2)) +
      xlab("Observed") +
      ylab("Predicted") +
      theme_classic() +
      labs(title = "In−Sample Predicted vs. Observed Values", 
           subtitle = paste("with 95% Confidence Intervals (actual coverage is",
                            paste0((1 - round(mean(df$outside), 2)) * 100, "%)"))) +
      theme(legend.position = "bottom") +
      ggpubr::grids(linetype = "dashed", color = "grey90")
  } else {
    m_range <- (df$up + df$low) / 2 # calculate the mean of each confidence interval
    df[, c("observed", "low", "up")] <- df[, c("observed", "low", "up")] - m_range # mean center the CIs and the observed values
    length <- df$up - df$low # calculate the length of CIs
    df <- df[order(length), ] # order the df in decreasing CIs length
    df$id <- 1:nrow(df)
    
    # Plot the mean centered CIs in decreasing order with the mean centered observations
    ggplot(df, aes(y = observed, x = id)) +
      geom_line(aes(y = up), linetype = 2, size = .5) +
      geom_line(aes(y = low),linetype = 2, size = .5) +
      geom_ribbon(aes(ymin = low, ymax = up), alpha = .15) +
      geom_point(aes(col = factor(outside)), alpha = alpha, size = 1) +
      scale_color_manual("", label = c("Inside", "Outside"),
                         values = ggsci::pal_lancet("lanonc")(2)) +
      ylab("observed values and prediction intervals (centered)") +
      theme_classic() +
      theme(axis.title.x = element_blank(),
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.position = "top")
  }
}


# 6.1 - Plotting graph of Observed vs Predicted values with bars of upper and lower prediction limits. 

# Observed values for ICT

obs.ICT <- ICT_NG[,3]

# Predicted mean values for ICT
pred.ICT <- Predict.ICT.output[,6]

# Binding Pred and Obs ICT to have same lenght 
Obs.Pred.ICT <- cbind(pred.ICT, obs.ICT) #extracting Obs ICT<->Pred ICT

# Now re-run the above commands to extract back the values.
obs.ICT <- Obs.Pred.ICT[,2]


# Predicted lower and upper bounds
pred025 <- Predict.ICT.output[,3]
pred975 <- Predict.ICT.output[,5]

ICT.CI <- as.data.frame(cbind(pred025, pred975))


install.packages(c("ggplot2", "ggpubr", "ggsci"))
library(ggplot2)
source("ggvalidate.R")

ggvalidate(observed = obs.ICT, predicted = pred.ICT, CI = cbind(pred025, pred975), type = 1, alpha = 1)

ggvalidate(observed = obs.ICT, predicted = pred.ICT, CI = cbind(pred025, pred975), type = 2, alpha = 1)

# Save workspace

# END OF RUN

