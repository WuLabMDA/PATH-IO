# delete work space
rm(list = ls(all = TRUE))
graphics.off()

# load useful libraries
library("beanplot")
library("biclust")
library("boot")
library("caret")
library("clusterRepro")
library("ConsensusClusterPlus")
library("dplyr")
library("EnhancedVolcano")
library("forestplot")
library("glmnet")
library("ggridges")
library("ggplot2")
library("Glimma")
library("igraph")
library("kernlab")
library("KRLS")
library("limma")
library("lubridate")
library("magrittr")
library("mclust")
library("survival")
library("survcomp")
library("survivalROC")
library("superheat")
library("Seurat")
library("randomForestSRC")
library("reshape2")
library("R.matlab")
library("Rtsne")
library("tsne")
library("survminer")
library("survival")

# set working directory

setwd("")

# load customized functions
source("concensus_clustering.R")
source("heatmap.3.R")
source("kmplot.R")
source("my_CC.R")

# Function to set a custom ggplot theme for KM plots
custom_theme <- function() {
  theme_survminer() %+replace%
    theme(
      plot.title = element_text(size = 14, color = "black", hjust = 0.5, face = "bold"),
      axis.text.x = element_text(size = 14, color = "black", face = "bold"),
      legend.text = element_text(size = 14, color = "black", face = "bold"),
      legend.title = element_text(size = 14, color = "black", face = "bold"),
      axis.text.y = element_text(size = 14, color = "black", face = "bold"),
      axis.title.x = element_text(size = 14, color = "black", face = "bold"),
      axis.title.y = element_text(size = 14, color = "black", face = "bold")
    )
}

#------------------------- Read Data -------------------------------
f.train <- read.csv("") 
f.test <- read.csv("") 

#------------------------- Risk Distribution Plot ------------------
plot(density(f.train$Risk_Prediction, bw = 5), lty = 2, lwd = 2, col = "red", 
     xlim = c(50, 180), ylim = c(0, 0.05), main = "Density Plots")
lines(density(f.test$Risk_Prediction, bw = 5), lty = 3, lwd = 2, col = "blue")
legend("topright", legend = c("Train", "Testing"), col = c("red", "blue"), 
       lty = c(2, 3), lwd = 3, cex = 1.2)

#------------------------- Identify Optimal Cutoff ------------------
HR.train <- rep(0, 19)
HR.test <- rep(0, 19)
p.train <- rep(1, 19)
p.test <- rep(1, 19)
c.off <- rep(1, 19)

for (i in 1:18) {
  cutoff <- quantile(f.train$Risk_Prediction, 0.05 * i)
  pt.stratify.train <- factor(f.train$Risk_Prediction >= cutoff)
  pt.stratify.test <- factor(f.test$Risk_Prediction >= cutoff)
  
  cox.train <- summary(coxph(Surv(f.train$OS, f.train$OS_Status) ~ pt.stratify.train))
  cox.test <- summary(coxph(Surv(f.test$OS, f.test$OS_Status) ~ pt.stratify.test))
  
  HR.train[i] <- cox.train$conf.int[2]
  p.train[i] <- cox.train$coefficients[5]
  HR.test[i] <- cox.test$conf.int[2]
  p.test[i] <- cox.test$coefficients[5]
  c.off[i] <- cutoff
}

print(cbind(HR.train, p.train, HR.test, p.test, c.off))

#------------------------- Select Cut-off ---------------------------
sele_cutoff <- 
cutoff_value <- quantile(f.train$Risk_Prediction, 0.05 * sele_cutoff)

pt.stratify.train <- factor(f.train$Risk_Prediction >= cutoff_value)
f.train$DeepFeatures <- factor(pt.stratify.train, levels = c("TRUE", "FALSE"))

#------------------------- Training KM Plot (Censored at 60 mo) -------------------------
# Apply censoring at 60 months
t <- 60
f.train$OS_c <- ifelse(f.train$OS > t, t, f.train$OS)
f.train$OS_Status_c <- ifelse(f.train$OS > t, 0, f.train$OS_Status)

pt.stratify.train <- factor(f.train$Risk_Prediction >= cutoff_value)
f.train$DeepFeatures <- factor(pt.stratify.train, levels = c("TRUE", "FALSE"))

cox_train <- coxph(Surv(OS_c, OS_Status_c) ~ pt.stratify.train, data = f.train)
cox_train_summary <- summary(cox_train)

HR_train <- round(cox_train_summary$conf.int[1, "exp(coef)"], 2)
lower_CI_train <- round(cox_train_summary$conf.int[1, "lower .95"], 2)
upper_CI_train <- round(cox_train_summary$conf.int[1, "upper .95"], 2)

hr_label_train <- paste0("HR: ", HR_train, " (95% CI: ", lower_CI_train, "-", upper_CI_train, ")")

fit <- survfit(Surv(OS_c, OS_Status_c) ~ DeepFeatures, data = f.train)

g <- ggsurvplot(
  fit,
  data = f.train,
  title = paste(hr_label_train),
  ggtheme = custom_theme(),
  conf.int = FALSE,
  pval = TRUE,                         # log-rank p-value
  fun = "pct",
  risk.table = TRUE,
  xlab = "Time (months)",
  ylab = "Overall Survival (%)",
  size = 1,
  risk.table.fontsize = 5,
  linetype = "strata",
  palette = c("Red", "Green4"),
  risk.table.col = "strata",
  legend.title = "Risk",
  legend.labs = c("High", "Low"),
  xlim = c(0, 60),                     # restrict to 60 months
  break.x.by = 6                       # every 6 months
)

# Adjust y-axis label and risk table font size
g$plot <- g$plot + theme(axis.title.y = element_text(angle = 90))
print(g)

#------------------------- Testing KM Plot (Censored at 60 mo) --------------------------
f.test$OS_c <- ifelse(f.test$OS > t, t, f.test$OS)
f.test$OS_Status_c <- ifelse(f.test$OS > t, 0, f.test$OS_Status)

pt.stratify.test <- factor(f.test$Risk_Prediction >= cutoff_value)
f.test$DeepFeatures <- factor(pt.stratify.test, levels = c("TRUE", "FALSE"))

cox_test <- coxph(Surv(OS_c, OS_Status_c) ~ pt.stratify.test, data = f.test)
cox_test_summary <- summary(cox_test)

HR_test <- round(cox_test_summary$conf.int[1, "exp(coef)"], 2)
lower_CI_test <- round(cox_test_summary$conf.int[1, "lower .95"], 2)
upper_CI_test <- round(cox_test_summary$conf.int[1, "upper .95"], 2)

hr_label_test <- paste0("HR: ", HR_test, " (95% CI: ", lower_CI_test, "-", upper_CI_test, ")")

fit <- survfit(Surv(OS_c, OS_Status_c) ~ DeepFeatures, data = f.test)

g <- ggsurvplot(
  fit,
  data = f.test,
  title = paste(hr_label_test),
  ggtheme = custom_theme(),
  conf.int = FALSE,
  pval = TRUE,                         # log-rank p-value
  fun = "pct",
  risk.table = TRUE,
  xlab = "Time (months)",
  ylab = "Overall Survival (%)",
  size = 1,
  risk.table.fontsize = 5,
  linetype = "strata",
  palette = c("Red", "Green4"),
  risk.table.col = "strata",
  legend.title = "Risk",
  legend.labs = c("High", "Low"),
  xlim = c(0, 60),                     # restrict to 60 months
  break.x.by = 6                       # every 6 months
)

# Adjust y-axis label and risk table font size
g$plot <- g$plot + theme(axis.title.y = element_text(angle = 90))
print(g)
