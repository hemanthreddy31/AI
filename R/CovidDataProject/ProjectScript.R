rm(list = ls())
library(Hmisc)


data <- read.csv("C:/R Programming/CovidDataProject/2019_nCoV_data.csv")
describe(data)
colnames(data)


data$death_dummy <- as.integer(data$Deaths != 0)
#death rate
sum(data$death_dummy)/nrow(data)

# AGE
# claim: people who die are older

dead = subset(data,death_dummy == 1)
alive = subset(data, death_dummy == 0)
describe(dead)
