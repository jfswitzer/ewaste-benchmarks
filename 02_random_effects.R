# Benchmark libraries
library(broom)
library(dplyr)
library(ggplot2)
library(scales)
library(data.table)
library(furrr)
library(tictoc)

# Setup multi-core parallel execution across available CPU cores
plan(multisession, workers = availableCores())

# Ensure output directory exists to prevent ggsave errors
dir.create("figures", showWarnings = FALSE)

topics <- c("crt_dei", "abortion", "cancel_culture", "court",  "environ", 
            "guns", "health", "imm", "lgbt", "police", "socialism")
mainstream <- c("Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "Threads")
alttech    <- c("Truth Social", "Telegram", "Rumble", "Gettr")
issue_groups <- list(
  culture_wars = c("crt_dei", "lgbt", "cancel_culture", "socialism"),
  hybrid       = c("abortion", "guns"),
  policy       = c("imm", "health", "court", "environ", "police")
)

tic("Total Execution Pipeline")

# 1. High-Performance Data Ingestion & Cleaning
tic("Data Ingestion & Cleaning")
df <- fread('all_posts_w_topic_and_candidate.csv')

# Light cleaning
df <- df[!is.na(likes) & Office.Name %in% c("U.S. Representative", "U.S. Senator")]

# Renaming platforms
platform_map <- c(
  "facebook" = "Facebook", "twitter" = "Twitter", "instagram" = "Instagram",
  "rumble" = "Rumble", "t" = "Telegram", "threads" = "Threads",
  "tiktok" = "TikTok", "truthsocial" = "Truth Social", "youtube" = "YouTube", 
  "gettr" = "Gettr"
)
df[, platform := recode(platform, !!!platform_map)]

# Fast vectorized row checks
df[, policy := as.integer(rowSums(.SD == 1, na.rm = TRUE) > 0), .SDcols = issue_groups$policy]
df[, hybrid := as.integer(rowSums(.SD == 1, na.rm = TRUE) > 0), .SDcols = issue_groups$hybrid]
df[, culture_wars := as.integer(rowSums(.SD == 1, na.rm = TRUE) > 0), .SDcols = issue_groups$culture_wars]

df[, issue_group := case_when(
  culture_wars == 1 ~ "culture wars",
  hybrid       == 1 ~ "hybrid",
  policy       == 1 ~ "policy issue",
  TRUE              ~ NA_character_
)]

df[, issue_group := factor(issue_group, levels = c("policy issue", "hybrid", "culture wars"))]

# Calculate values
df[, loglikes := log(likes + 2)]
df[, totalengagement := likes + shares]
df[, logtotal := log(totalengagement + 2)]
toc()

# 2. Parallel Model Fitting Across Cores
tic("Parallel Linear Regressions")

# Parallel group-by regression to test multi-core CPU throughput
all_results <- df %>%
  filter(!is.na(issue_group) & !is.na(Party.Standardized)) %>%
  group_by(Office.Name, Party.Standardized, platform) %>%
  filter(n() > 0) %>%
  nest() %>%
  future_pmap_dfr(function(Office.Name, Party.Standardized, platform, data) {
    result <- lm(loglikes ~ issue_group, data = data)
    
    tidy_result <- tidy(result) %>%
      mutate(
        platform = platform,
        party = Party.Standardized, # Fixed: Mapped from Party.Standardized correctly
        office = Office.Name
      )
    return(tidy_result)
  })

toc()

# 3. Post-Processing Coefficients
topic_results <- all_results %>%
  filter(term %in% c("issue_grouphybrid", "issue_groupculture wars"))

topic_results <- topic_results %>%
  mutate(
    estimate_likes   = exp(estimate) - 1,
    conf.low_likes   = exp(estimate - 1.96 * std.error) - 1,
    conf.high_likes  = exp(estimate + 1.96 * std.error) - 1
  )

# Renaming
topic_results$term <- recode(
  topic_results$term,
  "issue_grouphybrid" = "Hybrid",
  "issue_groupculture wars" = "Culture Wars"
)

topic_results$group <- ifelse(topic_results$platform %in% mainstream, "Mainstream", "Alt-tech")
topic_results$platform <- factor(topic_results$platform, levels = c(mainstream, alttech))

# 4. Figure Generation (Original Logic Preserved)
tic("Plot Rendering & File Output")

# Senate Plot
senate <- topic_results[topic_results$office == "U.S. Senator", ]

ggplot(senate, aes(x = platform, y = estimate_likes, color = term)) +
  geom_point(position = position_dodge(width = 0.6), size = 2) +
  geom_errorbar(
    aes(ymin = conf.low_likes, ymax = conf.high_likes),
    width = 0.2,
    position = position_dodge(width = 0.6)
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_flip() +
  theme_minimal() +
  facet_wrap(~ party, scales = "fixed") +
  labs(
    title = "Topic Effects on Likes (Senate)",
    x = "Platform",
    y = "Change in likes",
    color = "Issue Group"
  )

ggsave("figures/senate_topic_engagement_preds.pdf")

# House Plot
house <- topic_results[topic_results$office == "U.S. Representative", ]

ggplot(house, aes(x = platform, y = estimate, color = term)) +
  geom_point(position = position_dodge(width = 0.6), size = 2) +
  geom_errorbar(
    aes(ymin = estimate - 1.96 * std.error, ymax = estimate + 1.96 * std.error),
    width = 0.2,
    position = position_dodge(width = 0.6)
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_flip() +
  theme_minimal() +
  facet_wrap(~ party, scales = "fixed") +
  labs(
    title = "Topic Effects on Log Likes (House)",
    x = "Topic",
    y = "Coefficient Estimate (log likes)",
    color = "Platform"
  )

ggsave("figures/house_topic_engagement_preds.pdf")

toc() # End plot timing
toc() # End total execution timing