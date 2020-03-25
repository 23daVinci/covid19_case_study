# libraries
library(shiny)
library(shinyWidgets)
library(tidyr)
library(dplyr)
library(tidyr)
library(dplyr)
library(keras)
library(mlbench)
library(caret)

# ensure the results are repeatable
set.seed(7)

# loading the data
df1 <- read.csv("dataset/2019_nCoV_20200121_20200206.csv")
df2 <- read.csv("dataset/2019_nCoV_20200121_20200205.csv")
df3 <- read.csv("dataset/2019_nCoV_20200121_20200201.csv")
df4 <- read.csv("dataset/2019_nCoV_20200121_20200131.csv")
df5 <- read.csv("dataset/2019_nCoV_20200121_20200130.csv")
df6 <- read.csv("dataset/2019_nCoV_20200121_20200128.csv")
df7 <- read.csv("dataset/2019_nCoV_20200121_20200127.csv")

################# ANN preprocessing ###########################################

# dataset for ANN
dfANN <- read.csv("dataset/deepLearningDataset/COVID19_line_list_data.csv")
# removing unwanted data
dfANN <- subset(dfANN, select = -c(id, case_in_country, reporting.date, X, summary, If_onset_approximated, source, link))
dfANN <- subset(dfANN, select = -c(14:19))

# since the 'death' column also contains date values we need to clean them as well
deathCol <- dfANN$death
deathCol.factor <- factor(deathCol)
deathCol <- as.numeric(deathCol.factor)
deathCol[deathCol == 1] <- 0
deathCol[deathCol != 0] <- 1
dfANN$death <- deathCol

# Similarly normalizing the 'gender' column to binary (0 - male, 1 - female)
genderCol <- dfANN$gender
genderCol.factor <- factor(genderCol)
genderCol <- as.numeric(genderCol)
dfANN$gender <- genderCol

# Similarly normalizing the 'country' column
dfANN$country.factor <- factor(dfANN$country)
dfANN$country <- as.numeric(dfANN$country.factor)

# Similarly normalizing the 'symptom' column
dfANN$symptom.factor <- factor(dfANN$symptom)
dfANN$symptom <- as.numeric(dfANN$symptom.factor)

# Similarly normalizing the 'gender' column
genderCol <- dfANN$gender
genderCol.factor <- factor(genderCol)
genderCol <- as.numeric(genderCol)
dfANN$gender <- genderCol

colsToKeep <- c("country", "gender", "age", "symptom", "death")
corrDataset <- dfANN[colsToKeep]

# removing the rows with atleast one 'na'
corrDataset <- corrDataset[rowSums(is.na(corrDataset)) == 0,]

# dividing dataset into training and testing
## 75% of the sample size
smp_size <- floor(0.75 * nrow(corrDataset))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(corrDataset)), size = smp_size)

train <- corrDataset[train_ind, ]
test <- corrDataset[-train_ind, ]

# splitting into x and y
xTrain <- as.matrix(train[1:4])
yTrain <- as.matrix(train[5])
xTest <- as.matrix(test[1:4])
yTest <- as.matrix(test[5])

# One hot encoding
y_train <- to_categorical(yTrain)
y_test <- to_categorical(yTest)

# ANN model
# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 2, activation = 'softmax')

model = keras_model_sequential() %>%   
  layer_dense(units = 6, activation = "relu", input_shape = ncol(xTrain)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = ncol(y_train), activation = "softmax")

# compiling the model
compile(model, loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = "accuracy")

# fitting the model
history = fit(model,  xTrain, y_train, epochs = 20, batch_size = 128, validation_split = 0.2)

# saving the model
#saveRDS(model, file = "model.rds")

# loading the saved model
#loadedModel <- readRDS("model.rds")

# test data
#testDF <- data.frame(country = 9, gender = 2, age = 65, symptom = 5)
#testDf <- as.matrix(testDF)

# prediction
#y_data_pred=predict_classes(model, testDF)

#glimpse(y_data_pred)

########################################################################

# appending all the data frames into a single dataframe
dFrame <- rbind(df1, df2, df3, df4, df5, df6, df7)
# splitting Last.Update column into date and time
dFrame <- separate(data = dFrame, col = Last.Update, into = c("date", "time"), sep = "\\ ")



ui <- fluidPage(
  setBackgroundColor("black"),
  shinydashboard = TRUE,
  
  # title of the page
  withTags({
    div(class = "row container-fluid",
        h2(id = "title", "Coronavirus Global CaseStudy on John Hopkins CSSI data", style = "color:#adaaaa"),
        style = "background-color:#222222; margin-top:5px")
  }),
  
  # main content row
  fluidRow(
    # creating four columns (2, 6, 4)
    column(2,
           # creating row for total confirmed
           fluidRow(
             style = "background-color:#222222; margin-top:10px; margin-left:0px",
             h3("Total Confirmed", style = "color:white; text-align:center"),
             h1(textOutput("totalConfirmed"), style = "color:red; text-align:center; font-size:80px; font-weight:bold"), 
             
           ), 
           
           # creating row for country list
           fluidRow(
             style = "background-color:#222222; margin-top:10px; margin-left:0px",
             h5("Confirmed cases by country / region", style="color:#adaaaa; text-align:center; font-weight:bold"),
             br(),
             h6(" Select a country to view its details", style="color:#adaaaa; font-weight:bold"),
             # country list
             selectInput("country", "",
                         choices = c("World",
                                     "Mainland China",
                                     "Singapore",
                                     "Thailand",
                                     "Japan",
                                     "Hong Kong",
                                     "South Korea",
                                     "Germany",
                                     "Malaysia",
                                     "Taiwan",
                                     "Macau",
                                     "Vietnam",
                                     "France",
                                     "United Arab Emirates",
                                     "Australia",
                                     "India",
                                     "Canada",
                                     "Italy",
                                     "Philippines",
                                     "Russia",
                                     "UK",
                                     "United States",
                                     "Belgium",
                                     "Combodia",
                                     "Finland",
                                     "Nepal",
                                     "Spain",
                                     "Sri Lanka",
                                     "Sweden",
                                     "Ivory Coast",
                                     "Mexico",
                                     "Brazil",
                                     "Colombia"),
                         selected = "World"),
             
             br(),
             
             # city list
             h6(" Select a city to view its details", style="color:#adaaaa; font-weight:bold"),
             selectInput("city", "",
                         choices = c("all cities",
                         "Hubei",
                         "city two"),
                         selected = "all cities"),
             br(),
             
             # date list
             h6(" Select a date to filter the results", style="color:#adaaaa; font-weight:bold"),
             dateInput("dt", "", format = "mm/dd/yy"),
             br(), 
             br(),
           ),
           
           # creating a row for last updated date
           fluidRow(
             style = "background-color:#222222; margin-top:10px; margin-left:0px",
             h4("Last updated at (MM/DD/YYYY)", style="text-align:center; color:#aba9a9"),
             h3("23/02/2020 11:13:02 a.m.", style="text-align:center; color:#adaaaa; font-weight:bold")
           )
           ), 
    
    # map column
    column(7,
           # row for map and pre
           fluidRow(
             # column for map
             column(7,tabsetPanel(type = "tab",
                                   tabPanel("Map", style = "margin-top:10px",
                                            img(src="map.png", width="600px", height="480px", style="position:relative; top:-5px")),
                                   tabPanel("Prediction", style = "margin-top:10px; background-color:#222222",
                                            h2("Death prediction using ANN", style="color:white; text-align:center"),
                                            # creating a row for prediction
                                            fluidRow(
                                              # column for input form
                                              column(6, h3("Please fill the form to continue:", style = "color:#adaaaa; position:relative; left:10"),
                                                     # input for gender
                                                     selectInput("gender", "",
                                                                 choices = c("-- select gender --",
                                                                             "female",
                                                                             "male"
                                                                 ),
                                                                 selected = "-- select gender --"),
                                                     # input for age
                                                     p("Enter your age:", style = "color:#adaaaa"),
                                                     numericInput("age", "", 50, min = 1, max = 120),
                                                     # input for symptoms
                                                     selectInput("symptom", "",
                                                                 choices = c("-- select symtoms --",
                                                                             "none 1",
                                                                             "chest discomfort 2",
                                                                             "chills 3",
                                                                             "cold, fever, pneumonia 4",
                                                                             "cough 5",
                                                                             "cough with sputum 6",
                                                                             "cough, chest pain 7",
                                                                             "cough, chill, muscle pain 8",
                                                                             "cough, chills, jooint pain 9",
                                                                             "cough, chills, shortness of breath, diarrhea 10"
                                                                             ),
                                                                 selected = "-- select symptoms --"),
                                                     # input for country
                                                     selectInput("countryANN", "",
                                                                 choices = c("-- select country --",
                                                                             "Afghanistan 1",
                                                                             "Algeria 2",
                                                                             "Australia 3",
                                                                             "Austria 4",
                                                                             "Bahrain 5",
                                                                             "Belgium 6",
                                                                             "Cambodia 7",
                                                                             "Canada 8",
                                                                             "China 9",
                                                                             "Croatia 10"
                                                                 ),
                                                                 selected = "-- select country --")),
                                              # column for model output
                                              column(6, br(), h4("Probability of dying from COVID19 is:", style = "color:#adaaaa; position:relative; left:10"),
                                                     h1(textOutput("predValue"), "%", style = "color:orange; text-align:center; font-weight:bold"))
                                            ),
                                            br())
                                   )
                    ),
                    
             # column for pre
             column(5,
                    fluidRow(
                      br(),
                      style = "background-color:#222222; margin-top:10px",
                      h3("If the virus grows with this rate it will cover the entire planet in:", style="color:#adaaaa; font-weight:bold"),
                      h1(textOutput("spreadDays"), style = "color:orange; text-align:center; font-size:60px; font-weight:bold"),
                      br(),
                      br(),br()
                    ),
                    fluidRow(br(),
                      style = "background-color:#222222; margin-top:10px",
                      h3("If the treatment grows with this rate, the time neeeded to cure everyone will be:", style="color:#adaaaa; font-weight:bold"),
                      h1(textOutput("treatDays"), style = "color:blue; text-align:center; font-size:60px; font-weight:bold"),
                      br(),br()
                    )
                    )
           ),
           
           # row for growth graph
           fluidRow(
             style = "background-color:#222222; margin-top:10px; margin-left:0px",
             h4(" Confirmed cases vs Deaths vs Treated graph:", style="color:#adaaaa"),
             img(src="graph.png", width = "1000px", height = "298px")
           )
           ),
    
    # other two columns
    column(3,
           # creating upper row for two columns
           fluidRow(
             style = "background-color:#222222; margin-top:10px; margin-left:0px",
             h3("Total deaths", style = "color:white; text-align:center"),
             h1(textOutput("totalDeath"), style = "color:white; text-align:center; font-size:70px; font-weight:bold")
           ),
           fluidRow(
             style = "background-color:#222222; margin-top:10px; margin-left:0px",
             h3("Total Recovered", style = "color:white; text-align:center"),
             h1(textOutput("totalRecovered"), style = "color:green; text-align:center; font-size:70px; font-weight:bold")
           ),
           
           # row for top countries
           fluidRow(
             column(12,
                    style = "background-color:#222222; margin-top:10px; margin-left:14px",
                    tabsetPanel(type="tab",
                                tabPanel("Confirmed", h2("Top 6 countries", style = "color:white; text-align:center; font-weight:bold"),
                                         h3("Confirmed cases", style = "color:red; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("firstCountryNameConfirmed"), style = "color:red; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("secondCountryNameConfirmed"), style = "color:red; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("thirdCountryNameConfirmed"), style = "color:red; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("fourthCountryNameConfirmed"), style = "color:red; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("fifthCountryNameConfirmed"), style = "color:red; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("sixthCountryNameConfirmed"), style = "color:red; font-weight:bold"),
                                         hr()),
                                
                                tabPanel("Deaths", h2("Top 6 countries", style = "color:white; text-align:center; font-weight:bold"),
                                         h3("Death cases", style = "color:white; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("firstCountryNameDeath"), style = "color:white; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("secondCountryNameDeath"), style = "color:white; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("thirdCountryNameDeath"), style = "color:white; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("fourthCountryNameDeath"), style = "color:white; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("fifthCountryNameDeath"), style = "color:white; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("sixthCountryNameDeath"), style = "color:white; font-weight:bold"),
                                         hr()),
                                
                                tabPanel("Treated", h2("Top 6 countries", style = "color:white; text-align:center; font-weight:bold"),
                                         h3("Recovery cases", style = "color:Green; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("firstCountryNameRecovered"), style = "color:green; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("secondCountryNameRecovered"), style = "color:green; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("thirdCountryNameRecovered"), style = "color:green; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("fourthCountryNameRecovered"), style = "color:green; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("fifthCountryNameRecovered"), style = "color:green; font-weight:bold"),
                                         hr(),
                                         h4(textOutput("sixthCountryNameRecovered"), style = "color:green; font-weight:bold"),
                                         hr())
                    )
                    )
           ))
  ),
  
  # row for footer
  fluidRow(
    column(12,
           style = "background-color:#222222; margin-top:10px; margin-left:0px",
           h5("Created with love by Naman Lazarus, Monali Nayak, Janice Maria", style="text-align:center; color:white"))
  )
  
)

# server
server <- function(input, output) {
  
  # function for getting country number
  getCountryNumber <- function(countryName) {
    if (countryName == "Afghanistan 1") {
      return(1)
    }
    else if (countryName == "Algeria 2") {
      return(2)
    }
    else if (countryName == "Australia 3") {
      return(3)
    }
    else if (countryName == "Austria 4") {
      return(4)
    }
    else if (countryName == "Bahrain 5") {
      return(5)
    }
    else if (countryName == "Belgium 6") {
      return(6)
    }
    else if (countryName == "Cambodia 7") {
      return(7)
    }
    else if (countryName == "Canada 8") {
      return(8)
    }
    else if (countryName == "China 9") {
      return(9)
    }
    else if (countryName == "Croatia 10") {
      return(10)
    }
    
  }
  
  # function for getting symptom number
  getSymptomNumber <- function(symptomName) {
    if (symptomName == "none 1") {
      return(1)
    }
    else if (symptomName == "chest discomfort 2") {
      return(2)
    }
    else if (symptomName == "chills 3") {
      return(3)
    }
    else if (symptomName == "cold, fever, pneumonia 4") {
      return(4)
    }
    else if (symptomName == "cough 5") {
      return(5)
    }
    else if (symptomName == "cough with sputum 6") {
      return(6)
    }
    else if (symptomName == "cough, chest pain 7") {
      return(7)
    }
    else if (symptomName == "cough, chill, muscle pain 8") {
      return(8)
    }
    else if (symptomName == "cough, chills, joint pain 9") {
      return(9)
    }
    else if (symptomName == "cough, chills, shortness of breath, diarrhea 10") {
      return(10)
    }
    
  }
  
  # function to run prediction
  runPrediction <- function(cont, gen, ag, symp) {
    # createing xTest
    testDF <- data.frame(country = cont, gender = gen, age = ag, symptom = symp)
    testDF <- as.matrix(testDF)
    
    # prediction
    y_data_pred=predict_classes(model, testDF)
    #y_data_pred
    # one hot encoded prediction
    y_data_pred_oneh=predict(model, testDF) 
    dyingProb <- y_data_pred_oneh[1] * 100
    return(format(round(dyingProb, 2), nsmall = 2))
  }
  
  # function for correcting input date
  correctIpDate <- function() {
    # filtering dFrame based on input$dt
    # input$dt is of the form yyyy-mm-dd
    dtStr <- as.character(input$dt) # converting date to string
    dtList <- as.list(strsplit(dtStr, "-")) # splitting the string with delimeter '-'
    dtUnList <- unlist(dtList)[]
    # storing month, date and year separately
    yy <- dtUnList[1]
    mm <- dtUnList[2]
    dd <- dtUnList[3]
    
    # correcting date
    dtSplitted <- unlist(as.list(strsplit(dd, "")))[]
    if (dtSplitted[1] == 0) {
      dd <- dtSplitted[2]
    }
    
    # correcting month
    mmSplitted <- unlist(as.list(strsplit(mm, "")))[]
    if (mmSplitted[1] == 0) {
      mm <- mmSplitted[2]
    }
    
    res <- paste(mm, dd, yy, sep = "/")
    return(res)
  }
  
  # for ANN prediction
  output$predValue <- renderText({
    if ((input$countryANN != "-- select country --") && (input$gender != "-- select gender --") && (input$symptom != "-- select symptom --")) {
      # getting the corresponding number code from the given input text
      countryNum <- getCountryNumber(input$countryANN)
      genderNum = 0
      if (input$gender == "female") {
        genderNum = 1
      }
      else if (input$gender == "male") {
        genderNum = 2
      }
      symptomNum <- getSymptomNumber(input$symptom)
      
      # running prediction
      paste(runPrediction(countryNum, genderNum, input$age, symptomNum))
      
    }
    else {
      paste(" ")
    }
  })
  
  # getting selected values from the form
  # for confirmed
  output$totalConfirmed <- renderText({
    res <-  correctIpDate()
    
    dateFilteredDF <- subset(dFrame, date == res)
    if (input$city == "all cities") {
      # all cities
      if (input$country == "World") {
        paste(sum(dateFilteredDF$Confirmed, na.rm = TRUE))
      }
      else {
        # creating a filtered country dataframe
        countryDF <- subset(dateFilteredDF, Country.Region==input$country)
        paste(sum(countryDF$Confirmed, na.rm = TRUE))
      }
    }
    else {
      # some city is selected
      # creating a filtered city dataframe
      cityDF <- subset(dateFilteredDF, Province.State==input$city)
      paste(sum(cityDF$Confirmed, na.rm = TRUE))
    }
    
  })
  
  # for deaths
  output$totalDeath <- renderText({
    res <-  correctIpDate()
    
    dateFilteredDF <- subset(dFrame, date == res)
    if (input$city == "all cities") {
      # all cities
      if (input$country == "World") {
        paste(sum(dateFilteredDF$Death, na.rm = TRUE))
      }
      else {
        # creating a filtered country dataframe
        countryDF <- subset(dateFilteredDF, Country.Region==input$country)
        paste(sum(countryDF$Death, na.rm = TRUE))
      }
    }
    else {
      # some city is selected
      # creating a filtered city dataframe
      cityDF <- subset(dateFilteredDF, Province.State==input$city)
      paste(sum(cityDF$Death, na.rm = TRUE))
    }
    
  })
  
  # for recovered
  output$totalRecovered <- renderText({
    res <-  correctIpDate()
    
    dateFilteredDF <- subset(dFrame, date == res)
    if (input$city == "all cities") {
      # all cities
      if (input$country == "World") {
        paste(sum(dateFilteredDF$Recovered, na.rm = TRUE))
      }
      else {
        # creating a filtered country dataframe
        countryDF <- subset(dateFilteredDF, Country.Region==input$country)
        paste(sum(countryDF$Recovered, na.rm = TRUE))
      }
    }
    else {
      # some city is selected
      # creating a filtered city dataframe
      cityDF <- subset(dateFilteredDF, Province.State==input$city)
      paste(sum(cityDF$Recovered, na.rm = TRUE))
    }
    
  })
  
  # no of days required to spread
  output$spreadDays <- renderText({
    # getting min and max date from the dataset
    minDateLst <- sort(dFrame$date, decreasing = FALSE)
    minDate <- minDateLst[1]
    maxDate <- "2/3/20"
    
    # getting the number of confirmed on the earliest date
    cnfOnMinDateFilter <- subset(dFrame, date == minDate)
    cnfOnMinDate <- sum(cnfOnMinDateFilter$Confirmed, na.rm = TRUE)
    # getting the number of confirmed of the latest date
    cnfOnMaxDateFilter <- subset(dFrame, date == maxDate)
    cnfOnMaxDate <- sum(cnfOnMaxDateFilter$Confirmed, na.rm = TRUE)
    confirmRateOfGrowth <- (cnfOnMaxDate - cnfOnMinDate) / 14
    worldPopulation <- 7000000000
    daysToSpread <- worldPopulation / confirmRateOfGrowth
    daysToSpread <- daysToSpread - (daysToSpread * (87 / 100))
    paste(format(round(daysToSpread, 0), nsmall = 0))
  })
  
  # no of days required to treat
  output$treatDays <- renderText({
    # getting min and max date from the dataset
    minDateLst <- sort(dFrame$date, decreasing = FALSE)
    minDate <- minDateLst[1]
    maxDate <- "2/3/20"
    
    # getting the number of confirmed on the earliest date
    recoveredOnMinDateFilter <- subset(dFrame, date == minDate)
    recoveredOnMinDate <- sum(recoveredOnMinDateFilter$Recovered, na.rm = TRUE)
    # getting the number of confirmed of the latest date
    recoveredOnMaxDateFilter <- subset(dFrame, date == maxDate)
    recoveredOnMaxDate <- sum(recoveredOnMaxDateFilter$Recovered, na.rm = TRUE)
    recoveredRateOfGrowth <- (recoveredOnMaxDate - recoveredOnMinDate) / 14
    worldPopulation <- 700000000
    daysToTreat <- worldPopulation / recoveredRateOfGrowth
    daysToTreat <- daysToTreat - (daysToTreat * (90 / 100))
    paste(format(round(daysToTreat, 0), nsmall = 0))
  })
  
  # function to get top 6 Confirmed contries
  topSixConfirmedCountries <- function(ipDate) {
    # condition can be treated or deaths or confirmed
    topSixDF <- subset(dFrame, date == ipDate)
    uniqueCountryList <- unique(topSixDF$Country.Region)
    countryValueDF <- data.frame("country" = "sample", "cases" = 0) # creating an initial DF
    for (country in uniqueCountryList) {
      topSixDFWRTCountry <- subset(topSixDF, Country.Region == country)
      countryTotal <- sum(topSixDFWRTCountry$Confirmed, na.rm = TRUE)
      newDF <- data.frame("country" = country, "cases" = countryTotal) # creating a new DF to hold the row of the country
      countryValueDF <- rbind(countryValueDF, newDF)
    }
    # sorting the DF to be returned in decending order
    countryValueDF <- countryValueDF[order(-countryValueDF$cases),]
    return(countryValueDF)
  }
  
  # function to get top 6 Death contries
  topSixDeathCountries <- function(ipDate) {
    # condition can be treated or deaths or confirmed
    topSixDF <- subset(dFrame, date == ipDate)
    uniqueCountryList <- unique(topSixDF$Country.Region)
    countryValueDF <- data.frame("country" = "sample", "cases" = 0) # creating an initial DF
    for (country in uniqueCountryList) {
      topSixDFWRTCountry <- subset(topSixDF, Country.Region == country)
      countryTotal <- sum(topSixDFWRTCountry$Death, na.rm = TRUE)
      newDF <- data.frame("country" = country, "cases" = countryTotal) # creating a new DF to hold the row of the country
      countryValueDF <- rbind(countryValueDF, newDF)
    }
    # sorting the DF to be returned in decending order
    countryValueDF <- countryValueDF[order(-countryValueDF$cases),]
    return(countryValueDF)
  }
  
  # function to get top 6 Recovered contries
  topSixRecoveredCountries <- function(ipDate) {
    # condition can be treated or deaths or confirmed
    topSixDF <- subset(dFrame, date == ipDate)
    uniqueCountryList <- unique(topSixDF$Country.Region)
    countryValueDF <- data.frame("country" = "sample", "cases" = 0) # creating an initial DF
    for (country in uniqueCountryList) {
      topSixDFWRTCountry <- subset(topSixDF, Country.Region == country)
      countryTotal <- sum(topSixDFWRTCountry$Recovered, na.rm = TRUE)
      newDF <- data.frame("country" = country, "cases" = countryTotal) # creating a new DF to hold the row of the country
      countryValueDF <- rbind(countryValueDF, newDF)
    }
    # sorting the DF to be returned in decending order
    countryValueDF <- countryValueDF[order(-countryValueDF$cases),]
    return(countryValueDF)
  }
  
  # output for tabset
  # Confirmed
  # fist country
  output$firstCountryNameConfirmed <- renderText({
    res <-  correctIpDate()
    allDF <- topSixConfirmedCountries(res)
    paste(allDF[1,1], allDF[1,2])
  })
  
  # second country
  output$secondCountryNameConfirmed <- renderText({
    res <-  correctIpDate()
    allDF <- topSixConfirmedCountries(res)
    paste(allDF[2,1], allDF[2,2])
  })
  
  # third country
  output$thirdCountryNameConfirmed <- renderText({
    res <-  correctIpDate()
    allDF <- topSixConfirmedCountries(res)
    paste(allDF[3,1], allDF[3,2])
  })
  
  # fourth country
  output$fourthCountryNameConfirmed <- renderText({
    res <-  correctIpDate()
    allDF <- topSixConfirmedCountries(res)
    paste(allDF[4,1], allDF[4,2])
  })
  
  # fifth country
  output$fifthCountryNameConfirmed <- renderText({
    res <-  correctIpDate()
    allDF <- topSixConfirmedCountries(res)
    paste(allDF[5,1], allDF[5,2])
  })
  
  # sixth country
  output$sixthCountryNameConfirmed <- renderText({
    res <-  correctIpDate()
    allDF <- topSixConfirmedCountries(res)
    paste(allDF[6,1], allDF[6,2])
  })
  
  # Death
  # fist country
  output$firstCountryNameDeath <- renderText({
    res <-  correctIpDate()
    allDF <- topSixDeathCountries(res)
    paste(allDF[1,1], allDF[1,2])
  })
  
  # second country
  output$secondCountryNameDeath <- renderText({
    res <-  correctIpDate()
    allDF <- topSixDeathCountries(res)
    paste(allDF[2,1], allDF[2,2])
  })
  
  # third country
  output$thirdCountryNameDeath <- renderText({
    res <-  correctIpDate()
    allDF <- topSixDeathCountries(res)
    paste(allDF[3,1], allDF[3,2])
  })
  
  # fourth country
  output$fourthCountryNameDeath <- renderText({
    res <-  correctIpDate()
    allDF <- topSixDeathCountries(res)
    paste(allDF[4,1], allDF[4,2])
  })
  
  # fifth country
  output$fifthCountryNameDeath <- renderText({
    res <-  correctIpDate()
    allDF <- topSixDeathCountries(res)
    paste(allDF[5,1], allDF[5,2])
  })
  
  # sixth country
  output$sixthCountryNameDeath <- renderText({
    res <-  correctIpDate()
    allDF <- topSixDeathCountries(res)
    paste(allDF[6,1], allDF[6,2])
  })
  
  # Recovered
  # fist country
  output$firstCountryNameRecovered <- renderText({
    res <-  correctIpDate()
    allDF <- topSixRecoveredCountries(res)
    paste(allDF[1,1], allDF[1,2])
  })
  
  # second country
  output$secondCountryNameRecovered <- renderText({
    res <-  correctIpDate()
    allDF <- topSixRecoveredCountries(res)
    paste(allDF[2,1], allDF[2,2])
  })
  
  # third country
  output$thirdCountryNameRecovered <- renderText({
    res <-  correctIpDate()
    allDF <- topSixRecoveredCountries(res)
    paste(allDF[3,1], allDF[3,2])
  })
  
  # fourth country
  output$fourthCountryNameRecovered <- renderText({
    res <-  correctIpDate()
    allDF <- topSixRecoveredCountries(res)
    paste(allDF[4,1], allDF[4,2])
  })
  
  # fifth country
  output$fifthCountryNameRecovered <- renderText({
    res <-  correctIpDate()
    allDF <- topSixRecoveredCountries(res)
    paste(allDF[5,1], allDF[5,2])
  })
  
  # sixth country
  output$sixthCountryNameRecovered <- renderText({
    res <-  correctIpDate()
    allDF <- topSixRecoveredCountries(res)
    paste(allDF[6,1], allDF[6,2])
  })
  
}

# running the app
shinyApp(ui, server)