'''
MB Note: We still need to throw a random effect in there for office holder, but 
the nuts and bolts of the regression are here. 
'''
library(broom)
library(dplyr)
library(ggplot2)
library(scales)

topics <- c("crt_dei", "abortion", "cancel_culture", "court",  "environ", 
            "guns", "health", "imm", "lgbt", "police", "socialism")
mainstream <- c("Facebook", "Twitter", "Instagram", "TikTok", "YouTube", "Threads")
alttech    <- c("Truth Social", "Telegram", "Rumble", "Gettr")
issue_groups <- list(
  culture_wars = c("crt_dei", "lgbt", "cancel_culture", "socialism"),
  hybrid       = c("abortion", "guns"),
  policy       = c("imm", "health", "court", "environ", "police")
)



# read in data
df <- read.csv('all_posts_w_topic_and_candidate.csv')

# do some light cleaning
df <- df[!is.na(df$likes),]
df <- df[df$Office.Name == "U.S. Representative" | df$Office.Name == 'U.S. Senator',]

# renaming 
df$platform <- recode(df$platform,
                      "facebook" = "Facebook",
                      "twitter" = "Twitter",
                      "instagram" = "Instagram",
                      "rumble" = "Rumble",
                      "t" = "Telegram",
                      "threads" = "Threads",
                      "tiktok" = "TikTok",
                      "truthsocial" = "Truth Social",
                      "youtube" = "YouTube",
                      "gettr" = "Gettr"
)




df <- df %>% mutate(policy = if_any(all_of(issue_groups$policy), ~ . == 1) * 1L)
df <- df %>% mutate(hybrid = if_any(all_of(issue_groups$hybrid), ~ . == 1) * 1L)
df <- df %>% mutate(culture_wars = if_any(all_of(issue_groups$culture_wars), ~ . == 1) * 1L)

df <- df %>%
  mutate(
    issue_group = case_when(
      culture_wars == 1 ~ "culture wars",
      hybrid       == 1 ~ "hybrid",
      policy       == 1 ~ "policy issue",
      TRUE ~ NA_character_
    )
  )

df <- df %>%
  mutate(
    issue_group = factor(
      issue_group,
      levels = c("policy issue", "hybrid", "culture wars")
    )
  )

# calculate values
df$loglikes = log(df$likes + 2)
df$totalengagement = df$likes + df$shares 
df$logtotal <- log(df$totalengagement + 2)


platform_results <- list()


for (office in unique(df$Office.Name)){
  print(office)
  for (party in unique(df$Party.Standardized)){
    print(party)
    for (platform in unique(df$platform)){
      print(platform)
      platformdf <- df[df$platform == platform & df$Party.ID == party & df$Office.Name == office,]
      
      if (nrow(platformdf) > 0){
        result <- lm(loglikes ~ issue_group,
                     data=platformdf
        )
        
        # Store tidy coefficients with platform name
        tidy_result <- tidy(result) %>%
          mutate(platform = platform)
        
        tidy_result$party <- party
        tidy_result$office <- office
        
        platform_results[[paste0(platform, party, office)]] <- tidy_result
        
      }
      
    }
  }
}




all_results <- bind_rows(platform_results)

topic_results <- all_results %>%
  filter(term %in% c("issue_grouphybrid", "issue_groupculture wars"))

topic_results <- topic_results %>%
  mutate(
    estimate_likes   = exp(estimate) - 1,
    conf.low_likes   = exp(estimate - 1.96 * std.error) - 1,
    conf.high_likes  = exp(estimate + 1.96 * std.error) - 1
  )



# renaming 
topic_results$term <- recode(topic_results$term,
                             "issue_grouphybrid" = "Hybrid",
                             "issue_groupculture wars" = "Culture Wars"
)

topic_results$group <- ifelse(topic_results$platform %in% mainstream, "Mainstream", "Alt-tech")
topic_results$platform <- factor(topic_results$platform, levels = c(mainstream, alttech))

senate <- topic_results[topic_results$office == "U.S. Senator",]

ggplot(senate, aes(x = platform, y = estimate_likes, color = term)) +
  geom_point(position = position_dodge(width = 0.6), size = 2) +
  geom_errorbar(aes(ymin = conf.low_likes,
                    ymax = conf.high_likes),
                width = 0.2,
                position = position_dodge(width = 0.6)) +
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

house <- topic_results[topic_results$office == "U.S. Representative",]

ggplot(house, aes(x = platform, y = estimate, color = term)) +
  geom_point(position = position_dodge(width = 0.6), size = 2) +
  geom_errorbar(aes(ymin = estimate - 1.96 * std.error,
                    ymax = estimate + 1.96 * std.error),
                width = 0.2,
                position = position_dodge(width = 0.6)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_flip() +
  theme_minimal() +

  # facet only by party
  facet_wrap(~ party, scales = "fixed") +
  
  labs(
    title = "Topic Effects on Log Likes (House)",
    x = "Topic",
    y = "Coefficient Estimate (log likes)",
    color = "Platform"
  )

ggsave("figures/house_topic_engagement_preds.pdf")
