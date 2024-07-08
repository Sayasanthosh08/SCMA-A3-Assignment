# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(magrittr)

# Read the dataset
data <- read_csv("C:\\Users\\sayas\\OneDrive\\New folder\\python projects\\NSSO68.csv")

# Create a binary variable for non-vegetarian status using dplyr pipeline
data <- data %>%
  mutate(non_veg = case_when(
    eggsno_q > 0 ~ 1,
    fishprawn_q > 0 ~ 1,
    goatmeat_q > 0 ~ 1,
    beef_q > 0 ~ 1,
    pork_q > 0 ~ 1,
    chicken_q > 0 ~ 1,
    othrbirds_q > 0 ~ 1,
    TRUE ~ 0
  ))

# Select relevant variables for the probit model and handle missing values
data_clean <- data %>%
  select(non_veg, Age, Sex, hhdsz, Religion, Education, MPCE_URP, state, State_Region) %>%
  filter_all(all_vars(!is.na(.)))

# Convert categorical variables to factors
data_clean <- data_clean %>%
  mutate(
    Sex = as.factor(Sex),
    Religion = as.factor(Religion),
    state = as.factor(state),
    State_Region = as.factor(State_Region)
  )

# Fit the probit regression model using the glm function
probit_model <- glm(non_veg ~ Age + Sex + hhdsz + Religion + Education + MPCE_URP + state + State_Region, 
                    data = data_clean, family = binomial(link = "probit"))

# Summarize the model
summary(probit_model)

# Make predictions
data_clean <- data_clean %>%
  mutate(predicted_prob = predict(probit_model, type = "response"))

# Visualize the results
ggplot(data_clean, aes(x = predicted_prob, fill = as.factor(non_veg))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  labs(title = "Predicted Probability of Being Non-Vegetarian", x = "Predicted Probability", y = "Count") +
  scale_fill_manual(values = c("1" = "blue", "0" = "red"), name = "Non-Vegetarian Status", labels = c("No", "Yes"))

# Save the plot
ggsave("predicted_probabilities.png", width = 8, height = 6)

