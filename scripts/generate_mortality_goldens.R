#!/usr/bin/env Rscript
#
# Generate mortality golden files from R actuarial packages.
#
# Creates actuarial table from SOA 2012 IAM qx values to match Python/Julia.
#
# Requires:
#   - R 4.0+
#   - lifecontingencies package
#   - jsonlite package
#
# Usage:
#   Rscript scripts/generate_mortality_goldens.R
#
# Output:
#   tests/golden/outputs/mortality_r.json

library(jsonlite)
library(lifecontingencies)

# Output path - handle both direct invocation and sourcing
script_dir <- if (sys.nframe() > 0 && !is.null(sys.frame(1)$ofile)) {
  dirname(sys.frame(1)$ofile)
} else {
  "scripts"
}

output_path <- file.path(
  script_dir,
  "..",
  "tests",
  "golden",
  "outputs",
  "mortality_r.json"
)

# SOA 2012 IAM Period Table - Male qx values (from Julia MortalityTables.jl)
# These are the authoritative qx values from mort.soa.org
soa_2012_iam_male_qx <- c(
  0.001605, 0.000401, 0.000275, 0.000229, 0.000174, 0.000168, 0.000165,
  0.000159, 0.000143, 0.000129, 0.000113, 0.000111, 0.000132, 0.000169,
  0.000213, 0.000254, 0.000293, 0.000328, 0.000359, 0.000387, 0.000414,
  0.000443, 0.000473, 0.000513, 0.000554, 0.000602, 0.000655, 0.000688,
  0.000710, 0.000727, 0.000741, 0.000751, 0.000754, 0.000756, 0.000756,
  0.000756, 0.000756, 0.000756, 0.000756, 0.000800, 0.000859, 0.000926,
  0.000999, 0.001069, 0.001142, 0.001219, 0.001318, 0.001454, 0.001627,
  0.001829, 0.002057, 0.002302, 0.002545, 0.002779, 0.003011, 0.003254,
  0.003529, 0.003845, 0.004213, 0.004631, 0.005096, 0.005614, 0.006169,
  0.006759, 0.007398, 0.008106, 0.008548, 0.009076, 0.009708, 0.010463,
  0.011357, 0.012418, 0.013675, 0.015150, 0.016860, 0.018815, 0.021031,
  0.023540, 0.026375, 0.029572, 0.033234, 0.037533, 0.042261, 0.047441,
  0.053233, 0.059855, 0.067514, 0.076340, 0.086388, 0.097634, 0.109993,
  0.123119, 0.137168, 0.152171, 0.168194, 0.185260, 0.197322, 0.214751,
  0.232507, 0.250397, 0.268607, 0.290016, 0.311849, 0.333962, 0.356207,
  0.380000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000,
  0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000,
  0.400000, 1.000000
)

# SOA 2012 IAM Period Table - Female qx values
soa_2012_iam_female_qx <- c(
  0.001355, 0.000339, 0.000232, 0.000160, 0.000135, 0.000131, 0.000123,
  0.000113, 0.000104, 0.000096, 0.000091, 0.000090, 0.000097, 0.000110,
  0.000131, 0.000157, 0.000187, 0.000218, 0.000245, 0.000265, 0.000277,
  0.000283, 0.000286, 0.000290, 0.000295, 0.000302, 0.000311, 0.000320,
  0.000329, 0.000337, 0.000300, 0.000303, 0.000313, 0.000328, 0.000346,
  0.000367, 0.000390, 0.000417, 0.000448, 0.000486, 0.000552, 0.000606,
  0.000647, 0.000679, 0.000712, 0.000749, 0.000797, 0.000862, 0.000945,
  0.001046, 0.001161, 0.001289, 0.001422, 0.001555, 0.001691, 0.001837,
  0.001998, 0.002181, 0.002389, 0.002625, 0.003460, 0.003818, 0.004207,
  0.004629, 0.005088, 0.006146, 0.006578, 0.007052, 0.007576, 0.008165,
  0.009074, 0.009930, 0.010889, 0.011973, 0.013192, 0.014557, 0.016087,
  0.017817, 0.019797, 0.022072, 0.024821, 0.028152, 0.031948, 0.036251,
  0.041123, 0.046675, 0.053010, 0.060235, 0.068453, 0.077774, 0.088377,
  0.100310, 0.113499, 0.127956, 0.143761, 0.161013, 0.175088, 0.190531,
  0.206935, 0.224361, 0.242931, 0.262770, 0.284002, 0.306752, 0.331145,
  0.357313, 0.385389, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000,
  0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000,
  0.400000, 1.000000
)

# Convert qx values to lx (survivors) starting from l0 = 100000
qx_to_lx <- function(qx_vec, l0 = 100000) {
  lx <- numeric(length(qx_vec) + 1)
  lx[1] <- l0
  for (i in seq_along(qx_vec)) {
    lx[i + 1] <- lx[i] * (1 - qx_vec[i])
  }
  return(lx)
}

# Create actuarial table from qx values
create_actuarial_table <- function(qx_vec, name = "SOA_2012_IAM") {
  lx <- qx_to_lx(qx_vec)
  # Remove the terminal lx (it would be 0)
  lx <- lx[-length(lx)]
  # Create actuarial table with 5% interest rate as default
  tbl <- new("actuarialtable",
    x = 0:(length(lx) - 1),
    lx = lx,
    interest = 0.05,
    name = name
  )
  return(tbl)
}

# Get qx values for specified ages
get_qx_values <- function(table, ages) {
  qx_list <- list()
  for (age in ages) {
    tryCatch({
      # qxt gives q_x,t (probability of dying in next t years)
      qx_val <- qxt(table, x = age, t = 1)
      qx_list[[as.character(age)]] <- qx_val
    }, error = function(e) {
      message(sprintf("Could not get qx for age %d: %s", age, e$message))
    })
  }
  return(qx_list)
}

# Calculate annuity factor ax (life annuity due)
calculate_ax <- function(table, age, rate) {
  tryCatch({
    return(axn(actuarialtable = table, x = age, i = rate))
  }, error = function(e) {
    message(sprintf("Could not calculate ax: %s", e$message))
    return(NULL)
  })
}

# Calculate whole life insurance Ax
calculate_Ax <- function(table, age, rate) {
  tryCatch({
    return(Axn(actuarialtable = table, x = age, i = rate))
  }, error = function(e) {
    message(sprintf("Could not calculate Ax: %s", e$message))
    return(NULL)
  })
}

main <- function() {
  message("Generating mortality golden files from R...")

  # Create SOA 2012 IAM tables from qx values
  message("Creating SOA 2012 IAM Male table...")
  male_table <- create_actuarial_table(soa_2012_iam_male_qx, "SOA_2012_IAM_Male")

  message("Creating SOA 2012 IAM Female table...")
  female_table <- create_actuarial_table(soa_2012_iam_female_qx, "SOA_2012_IAM_Female")

  # Test ages
  ages <- c(30, 40, 50, 60, 65, 70, 80, 90)

  # Build golden data
  golden <- list(
    "_meta" = list(
      source = "R lifecontingencies package with SOA 2012 IAM qx values",
      r_version = paste(R.version$major, R.version$minor, sep = "."),
      lifecontingencies_version = as.character(packageVersion("lifecontingencies")),
      generated = format(Sys.Date(), "%Y-%m-%d"),
      notes = "Actuarial table constructed from SOA 2012 IAM Period qx values"
    ),
    soa_2012_iam_male = list(
      qx = get_qx_values(male_table, ages),
      ax_65_i5 = calculate_ax(male_table, 65, 0.05),
      Ax_65_i5 = calculate_Ax(male_table, 65, 0.05),
      ax_65_i3 = calculate_ax(male_table, 65, 0.03),
      ax_70_i5 = calculate_ax(male_table, 70, 0.05)
    ),
    soa_2012_iam_female = list(
      qx = get_qx_values(female_table, ages),
      ax_65_i5 = calculate_ax(female_table, 65, 0.05)
    )
  )

  # Write to file
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  write_json(golden, output_path, pretty = TRUE, auto_unbox = TRUE)

  message(sprintf("Written: %s", output_path))
  message("Done!")
}

# Run main
main()
