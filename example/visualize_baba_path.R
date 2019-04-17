library(shiny)
library(tidyr)
library(dplyr)
library(ggplot2)
library(ape)
library(jsonlite)
library(gplots)
library(phytools)

get_limits <- function(vec){
  c(min(vec), max(vec))
}

plot_baba_components <- function(inferredComponents, sparsityLvl){
  sparsity_levels = unique(inferredComponents$sparsity)
  sparsity_idx <- which.min(abs(sparsityLvl - sparsity_levels))

  sparsityLvl <- sparsity_levels[sparsity_idx]

  inferredComponents %>%
    filter(sparsity == sparsityLvl) ->
    currComponents

  if (max(currComponents$ComponentWeight) > 0) {
    currComponents %>%
      filter(ComponentWeight > 0) ->
      currComponents
  }

  currComponents %>%
    ggplot(aes(x=Component, label=Population,
               y=Population, size=PopulationWeight*ComponentWeight,
               color=ComponentWeight, alpha=PopulationWeight)) +
    geom_text() +
    scale_x_continuous(expand=c(.1,0)) +
    scale_alpha_continuous(limits = get_limits(inferredComponents$PopulationWeight)) +
    scale_size_continuous(limits=get_limits(inferredComponents$PopulationWeight *
                                        inferredComponents$ComponentWeight)) +
    scale_color_continuous(limits=get_limits(inferredComponents$ComponentWeight),
                          low="blue", high="red") +
    facet_grid(Leaf ~ .) +
    ggtitle(paste("sparsity = ", sparsityLvl))
}

read_baba_path <- function(dirname) {
  fnames <- list.files(dirname, pattern=".*_decomposition.txt")
  do.call(rbind, lapply(paste(dirname, fnames, sep="/"), read.table, head=T))
}

shiny_baba_path <- function(inferredComponents, logStep) {
  sparsity_levels = unique(inferredComponents$sparsity)
  if (!logStep) {
    minSparsity = round(min(sparsity_levels), 2)
    maxSparsity = round(max(sparsity_levels), 2)
    sparsity_step = round((max(sparsity_levels) - min(sparsity_levels)) / (length(sparsity_levels)-1), 2)
  } else {
    log_sparsity_levels <- log10(sparsity_levels)
    min_logSparsity = round(min(log_sparsity_levels), 2)
    max_logSparsity = round(max(log_sparsity_levels), 2)
    logSparsity_step = round((max(log_sparsity_levels) -
                              min(log_sparsity_levels)) /
                             (length(log_sparsity_levels)-1), 2)
  }

  plotsBySparsLvl <- lapply(sparsity_levels, function(sparsLvl) {
    inferredComponents %>%
      plot_baba_components(sparsLvl) +
    theme(legend.position="bottom", text=element_text(size=16),
          axis.text=element_text(size=10))
  })

  if (logStep) {
    slider <- sliderInput(
      "logSparsity",
      "logSparsity =",
      min = min_logSparsity,
      max = max_logSparsity,
      step = logSparsity_step,
      value = mean(c(min_logSparsity, max_logSparsity)))
  } else {
    slider <- sliderInput("sparsity",
                "sparsity =",
                min = minSparsity,
                max = maxSparsity,
                step = sparsity_step,
                value = mean(c(minSparsity, maxSparsity)))
  }

  ui <- fluidPage(
    #titlePanel("sparse quartet decomposition"),
    slider, plotOutput("distPlot", height = "600px")
  )

  server <- function(input, output) {
    output$distPlot <- renderPlot({
      # find the nearest sparsity level
      if (logStep) {
      idx = which.min(abs(input$logSparsity - log_sparsity_levels))
      } else {
      idx = which.min(abs(input$sparsity - sparsity_levels))
      }
      plotsBySparsLvl[[idx]]
    })
  }

  # Run the application
  shinyApp(ui = ui, server = server)
}

inferredComponents <- read_baba_path("fit_baba_path_results")
shiny_baba_path(inferredComponents, logStep=TRUE)

plot_baba_components(inferredComponents, 630) +
  theme(legend.position="bottom", text=element_text(size=16),
        title=element_text(size=10), axis.text=element_text(size=8), legend.text=element_text(size=10),
        legend.key.width=grid::unit(2, "line"))

ggsave("baba_path_raw.png")
