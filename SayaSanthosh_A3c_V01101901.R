# Load necessary libraries
library(survival)  # For Tobit regression
library(readr)     # For reading CSV files

# Load the dataset
data <- read_csv("C:\\Users\\sayas\\OneDrive\\New folder\\python projects\\NSSO68.csv")

# Inspect the dataset
head(data)

# Selecting relevant columns for analysis
selected_cols <- c("MPCE_URP", "Age", "Sex", "Education", "Religion", "hhdsz")
data_selected <- data[selected_cols]

# Handling missing values if any
data_selected <- na.omit(data_selected)

# Convert categorical variables to factors
data_selected$Sex <- as.factor(data_selected$Sex)
data_selected$Religion <- as.factor(data_selected$Religion)
data_selected$Education <- as.factor(data_selected$Education)

# Perform Tobit regression using survreg (Tobit model)
# Assume left-censoring at 0 for MPCE_URP
tobit_model <- survreg(Surv(pmax(MPCE_URP, 0)) ~ Age + Sex + Education + Religion + hhdsz, 
                       data = data_selected, dist = "gaussian")

# Summary of the Tobit model
summary(tobit_model)

