using Documenter, ProxCox

makedocs(;
    modules=[ProxCox],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/kose-y/ProxCox.jl/blob/{commit}{path}#L{line}",
    sitename="ProxCox.jl",
    authors="Seyoon",
    assets=String[],
)

deploydocs(;
    repo="github.com/kose-y/ProxCox.jl",
)
