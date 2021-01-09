using Documenter, ParProx

makedocs(;
    modules=[ParProx],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/kose-y/ProxCox.jl/blob/{commit}{path}#L{line}",
    sitename="ParProx.jl",
    authors="Seyoon Ko",
    assets=String[],
)

deploydocs(;
    repo="github.com/kose-y/ParProx.jl",
)
