require'pl.path'

local opt = {
    {
        dataPath = 'data/imdb-fix.lua',
        dataMask = {tr=false, val=false, te=true},

        envPath = path.join('cv/imdb-randchar',
            "seqLength887-HU190-cv7dr-cv5dr-cv3dr-fc-o_epoch12.00_lossval0.3718.t7"
        ),

        seqLength = 887,
        batSize = 250,
    },

    {
        dataPath = 'data/imdb-fix.lua',
        dataMask = {tr=false, val=false, te=true},

        envPath = path.join('cv/imdb-randchar',
            "seqLength887-HU1000-cv24max-o_epoch9.00_lossval0.3811.t7"
        ),

        seqLength = 887,
        batSize = 250,
    }
}

dofile('testvote.lua').main(opt)