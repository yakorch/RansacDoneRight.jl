using RansacDoneRight
using Documenter

DocMeta.setdocmeta!(RansacDoneRight, :DocTestSetup, :(using RansacDoneRight); recursive=true)

makedocs(;
    modules=[RansacDoneRight],
    authors="yakorch <korchyar19@gmail.com>, KushnirDmytro <kushnir.dmytro.93@gmail.com>, prittjam <jbpritts@gmail.com>",
    sitename="RansacDoneRight.jl",
    format=Documenter.HTML(;
        canonical="https://yakorch.github.io/RansacDoneRight.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yakorch/RansacDoneRight.jl",
    devbranch="main",
)
