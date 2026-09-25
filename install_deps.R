# install_deps.R

required_packages <- c(
  "broom",
  "dplyr",
  "ggplot2",
  "scales",
  "data.table",
  "future",
  "furrr",
  "purrr",
  "tictoc"
)

new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

if (length(new_packages) > 0) {
  install.packages(new_packages, repos = "https://cloud.r-project.org")
}