#!/usr/bin/env julia
"""
Generate mortality golden files from Julia actuarial packages.

Requires:
    - Julia 1.9+
    - MortalityTables.jl
    - LifeContingencies.jl

Usage:
    julia scripts/generate_mortality_goldens.jl

Output:
    tests/golden/outputs/mortality_julia.json
"""

using Pkg

# Ensure packages are available
try
    using MortalityTables
    using LifeContingencies
    using JSON
catch e
    @warn "Installing required packages..."
    Pkg.add(["MortalityTables", "LifeContingencies", "JSON"])
    using MortalityTables
    using LifeContingencies
    using JSON
end

using Dates

# Output path
const OUTPUT_PATH = joinpath(@__DIR__, "..", "tests", "golden", "outputs", "mortality_julia.json")

"""
Generate qx values for a mortality table at specified ages.

Note: MortalityTables.jl uses direct indexing (table[age]) to get qx values.
"""
function get_qx_values(table, ages)
    qx_dict = Dict{String,Any}()
    for age in ages
        try
            # MortalityTables.jl uses direct indexing: table[age] returns qx
            qx = table[age]
            qx_dict[string(age)] = qx
        catch e
            @warn "Could not get qx for age $age: $e"
        end
    end
    return qx_dict
end

"""
Calculate life annuity-due factor ax using LifeContingencies.

Note: LifeContingencies v2.5+ uses AnnuityDue struct with present_value function.
"""
function calculate_ax(table, age, rate)
    try
        # Use table.ultimate for the full mortality vector
        life = SingleLife(mortality = table.ultimate, issue_age = age)
        annuity = AnnuityDue(life, rate)
        return present_value(annuity)
    catch e
        @warn "Could not calculate ax for age $age at rate $rate: $e"
        return nothing
    end
end

"""
Calculate whole life insurance Ax using LifeContingencies.
"""
function calculate_Ax(table, age, rate)
    try
        life = SingleLife(mortality = table.ultimate, issue_age = age)
        ins = Insurance(life, rate)
        return present_value(ins)
    catch e
        @warn "Could not calculate Ax for age $age at rate $rate: $e"
        return nothing
    end
end

function main()
    println("Generating mortality golden files from Julia...")

    # Load SOA tables using get_SOA_table with official table names
    # Note: Use exact names as they appear in mort.soa.org
    println("Loading SOA 2012 IAM tables...")
    soa_2012_iam_male = get_SOA_table("2012 IAM Period Table – Male, ANB")
    soa_2012_iam_female = get_SOA_table("2012 IAM Period Table – Female, ANB")

    # Test ages
    ages = [30, 40, 50, 60, 65, 70, 80, 90]

    # Build golden data
    golden = Dict{String,Any}(
        "_meta" => Dict(
            "source" => "MortalityTables.jl + LifeContingencies.jl",
            "julia_version" => string(VERSION),
            "generated" => Dates.format(now(), "yyyy-mm-dd"),
            "notes" => "Generated from SOA mortality tables via Julia actuarial packages"
        ),
        "soa_2012_iam_male" => Dict(
            "qx" => get_qx_values(soa_2012_iam_male, ages),
            "ax_65_i5" => calculate_ax(soa_2012_iam_male, 65, 0.05),
            "Ax_65_i5" => calculate_Ax(soa_2012_iam_male, 65, 0.05),
            "ax_65_i3" => calculate_ax(soa_2012_iam_male, 65, 0.03),
            "ax_70_i5" => calculate_ax(soa_2012_iam_male, 70, 0.05)
        ),
        "soa_2012_iam_female" => Dict(
            "qx" => get_qx_values(soa_2012_iam_female, ages),
            "ax_65_i5" => calculate_ax(soa_2012_iam_female, 65, 0.05)
        )
    )

    # Write to file
    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do io
        JSON.print(io, golden, 2)
    end

    println("Written: $OUTPUT_PATH")
    println("Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
