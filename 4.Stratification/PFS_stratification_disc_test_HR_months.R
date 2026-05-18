# ============================================================
# Clear workspace
# ============================================================

rm(list = ls(all = TRUE))
graphics.off()

# ============================================================
# Load libraries
# ============================================================

library(survival)
library(survminer)
library(ggplot2)

# ============================================================
# Custom KM plot theme
# ============================================================

custom_theme <- function() {

  theme_survminer() %+replace%
    theme(
      plot.title = element_text(
        size = 14,
        color = "black",
        hjust = 0.5,
        face = "bold"
      ),

      axis.text.x = element_text(
        size = 14,
        color = "black",
        face = "bold"
      ),

      axis.text.y = element_text(
        size = 14,
        color = "black",
        face = "bold"
      ),

      axis.title.x = element_text(
        size = 14,
        color = "black",
        face = "bold"
      ),

      axis.title.y = element_text(
        size = 14,
        color = "black",
        face = "bold"
      ),

      legend.text = element_text(
        size = 14,
        color = "black",
        face = "bold"
      ),

      legend.title = element_text(
        size = 14,
        color = "black",
        face = "bold"
      )
    )
}


# ============================================================
# User-defined paths
# ============================================================

train_csv <- "/path/to/train_risk_predictions.csv"

test_csv <- "/path/to/test_risk_predictions.csv"


# ============================================================
# Read data
# ============================================================

f.train <- read.csv(train_csv)

f.test <- read.csv(test_csv)


# ============================================================
# Risk distribution plot
# ============================================================

plot(
  density(f.train$Risk_Prediction, bw = 5),
  lty = 2,
  lwd = 2,
  col = "red",
  xlim = c(50, 180),
  ylim = c(0, 0.05),
  main = "Risk Score Density Plot"
)

lines(
  density(f.test$Risk_Prediction, bw = 5),
  lty = 3,
  lwd = 2,
  col = "blue"
)

legend(
  "topright",
  legend = c("Train", "Test"),
  col = c("red", "blue"),
  lty = c(2, 3),
  lwd = 3,
  cex = 1.2
)


# ============================================================
# Identify optimal cutoff
# ============================================================

HR.train <- rep(0, 19)
HR.test <- rep(0, 19)

p.train <- rep(1, 19)
p.test <- rep(1, 19)

c.off <- rep(1, 19)

for (i in 1:18) {

  cutoff <- quantile(
    f.train$Risk_Prediction,
    0.05 * i
  )

  pt.stratify.train <- factor(
    f.train$Risk_Prediction >= cutoff
  )

  pt.stratify.test <- factor(
    f.test$Risk_Prediction >= cutoff
  )

  cox.train <- summary(
    coxph(
      Surv(PFS, PFS_Status) ~ pt.stratify.train,
      data = f.train
    )
  )

  cox.test <- summary(
    coxph(
      Surv(PFS, PFS_Status) ~ pt.stratify.test,
      data = f.test
    )
  )

  HR.train[i] <- cox.train$conf.int[2]
  p.train[i] <- cox.train$coefficients[5]

  HR.test[i] <- cox.test$conf.int[2]
  p.test[i] <- cox.test$coefficients[5]

  c.off[i] <- cutoff
}


cutoff_results <- data.frame(
  Percentile = seq(0.05, 0.90, by = 0.05),
  Cutoff = c.off[1:18],
  HR_Train = HR.train[1:18],
  P_Train = p.train[1:18],
  HR_Test = HR.test[1:18],
  P_Test = p.test[1:18]
)

print(cutoff_results)


# ============================================================
# Select cutoff percentile
# ============================================================

sele_cutoff <- 10
# Example:
# 10 means 50th percentile
# because 0.05 * 10 = 0.50

cutoff_value <- quantile(
  f.train$Risk_Prediction,
  0.05 * sele_cutoff
)

cat("\nSelected cutoff:", cutoff_value, "\n")


# ============================================================
# Apply 60-month censoring
# ============================================================

t <- 60

f.train$PFS_c <- ifelse(
  f.train$PFS > t,
  t,
  f.train$PFS
)

f.train$PFS_Status_c <- ifelse(
  f.train$PFS > t,
  0,
  f.train$PFS_Status
)

f.test$PFS_c <- ifelse(
  f.test$PFS > t,
  t,
  f.test$PFS
)

f.test$PFS_Status_c <- ifelse(
  f.test$PFS > t,
  0,
  f.test$PFS_Status
)


# ============================================================
# Function to generate KM plot
# ============================================================

generate_km_plot <- function(data_df, title_text) {

  pt.stratify <- factor(
    data_df$Risk_Prediction >= cutoff_value
  )

  data_df$DeepFeatures <- factor(
    pt.stratify,
    levels = c("TRUE", "FALSE")
  )

  cox_model <- coxph(
    Surv(PFS_c, PFS_Status_c) ~ pt.stratify,
    data = data_df
  )

  cox_summary <- summary(cox_model)

  HR <- round(
    cox_summary$conf.int[1, "exp(coef)"],
    2
  )

  lower_CI <- round(
    cox_summary$conf.int[1, "lower .95"],
    2
  )

  upper_CI <- round(
    cox_summary$conf.int[1, "upper .95"],
    2
  )

  hr_label <- paste0(
    "HR: ",
    HR,
    " (95% CI: ",
    lower_CI,
    "-",
    upper_CI,
    ")"
  )

  fit <- survfit(
    Surv(PFS_c, PFS_Status_c) ~ DeepFeatures,
    data = data_df
  )

  g <- ggsurvplot(
    fit,
    data = data_df,

    title = hr_label,

    ggtheme = custom_theme(),

    conf.int = FALSE,
    pval = TRUE,

    fun = "pct",

    risk.table = TRUE,

    xlab = "Time (months)",
    ylab = "Progression-free Survival (%)",

    size = 1,

    risk.table.fontsize = 5,

    linetype = "strata",

    palette = c("Red", "Green4"),

    risk.table.col = "strata",

    legend.title = "Risk",
    legend.labs = c("High", "Low"),

    xlim = c(0, 60),

    break.x.by = 6
  )

  g$plot <- g$plot +
    theme(
      axis.title.y = element_text(angle = 90)
    )

  print(g)
}


# ============================================================
# Training KM plot
# ============================================================

generate_km_plot(
  f.train,
  "Training Cohort"
)


# ============================================================
# Testing KM plot
# ============================================================

generate_km_plot(
  f.test,
  "Testing Cohort"
)
