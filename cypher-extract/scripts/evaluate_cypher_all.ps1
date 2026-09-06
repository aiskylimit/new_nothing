[CmdletBinding()]
param(
    [string]$InferenceRoot = "results/inference/qwen3",
    [string]$EvaluationRoot = "results/evaluation/qwen3",
    [int[]]$Seeds = @(10, 42, 50, 100, 1234),
    [string[]]$Methods = @(),
    [string[]]$Datasets = @("cypherbench", "mind_the_query", "neo4j_text2cypher"),
    [string[]]$Metrics = @("execution_accuracy", "psjs", "executable"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $RepositoryRoot "src"
if (-not [IO.Path]::IsPathRooted($InferenceRoot)) {
    $InferenceRoot = Join-Path $RepositoryRoot $InferenceRoot
}
if (-not [IO.Path]::IsPathRooted($EvaluationRoot)) {
    $EvaluationRoot = Join-Path $RepositoryRoot $EvaluationRoot
}
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$SourceRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
} else {
    $SourceRoot
}

$DatasetGraphs = @{
    cypherbench = @(
        "company", "fictional_character", "flight_accident", "geography", "movie", "nba", "politics"
    )
    mind_the_query = @("bloom50", "healthcare", "wwc")
    neo4j_text2cypher = @(
        "bluesky", "buzzoverflow", "companies", "fincen", "gameofthrones", "grandstack", "movies",
        "neoflix", "network", "northwind", "offshoreleaks", "recommendations", "stackoverflow2", "twitch",
        "twitter"
    )
}

$DatasetConnectors = @{
    cypherbench = "cypherbench-db"
    mind_the_query = "mind-the-query-db"
    neo4j_text2cypher = "neo4j_text2cypher_db"
}

foreach ($dataset in $Datasets) {
    if (-not $DatasetGraphs.ContainsKey($dataset)) {
        throw "Unknown dataset '$dataset'."
    }
}

foreach ($seed in $Seeds) {
    $seedInput = Join-Path $InferenceRoot "seed$seed"
    if (-not (Test-Path -LiteralPath $seedInput -PathType Container)) {
        throw "Inference seed directory does not exist: $seedInput"
    }

    $seedMethods = $Methods
    if ($seedMethods.Count -eq 0) {
        $seedMethods = @(
            Get-ChildItem -LiteralPath $seedInput -Directory |
                Sort-Object Name |
                Select-Object -ExpandProperty Name
        )
    }
    if ($seedMethods.Count -eq 0) {
        throw "No method directories found under $seedInput"
    }

    Write-Host "[seed$seed] evaluating methods: $($seedMethods -join ', ')"
    foreach ($method in $seedMethods) {
        foreach ($dataset in $Datasets) {
            $inputPath = Join-Path $seedInput "$method/$dataset/generator_predictions.jsonl"
            if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
                throw "Missing inference output: $inputPath"
            }

            $datasetOutput = Join-Path $EvaluationRoot "seed$seed/$method/$dataset"
            foreach ($graph in $DatasetGraphs[$dataset]) {
                $outputPath = Join-Path $datasetOutput "$graph/cypher_scores.jsonl"
                $evalArgs = @(
                    "-m", "cypher_evaluation.cli",
                    "--input", $inputPath,
                    "--output", $outputPath,
                    "--name", $DatasetConnectors[$dataset],
                    "--graph", $graph,
                    "--metrics"
                ) + $Metrics
                Write-Host "[seed$seed/$method/$dataset/$graph] evaluating"
                & $Python @evalArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Evaluation failed for seed$seed/$method/$dataset/$graph (exit $LASTEXITCODE)."
                }
            }

            $mergeArgs = @(
                "-m", "cypher_evaluation.merge",
                "--input-dir", $datasetOutput,
                "--metrics"
            ) + $Metrics
            & $Python @mergeArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Merge failed for seed$seed/$method/$dataset (exit $LASTEXITCODE)."
            }
        }
    }
    Write-Host "[seed$seed] complete"
}
