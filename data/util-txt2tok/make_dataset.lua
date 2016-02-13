--- make imdb dataset, use pre-defined lower-case char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

--- global config: pl script and registerd-token
local FN_REGISTER = path.join(
    paths.cwd(), 'data', 'util-txt2tok',
    'registered-tokens.txt'
)
local PL_SCRIPT = path.join(
    paths.cwd(), 'data', 'util-txt2tok',
    'to_tokens.pl'
)

--- global config: dbpedia train
--local PATH_DATA = '/home/ps/data/dbpedia/tok-cat'
--local FN_TXT = 'train.txt'

--- global config: dbpedia train, deepml
local PATH_DATA = '/mnt/data/datasets/Text/dbpedia/tok-cat'
local FN_TXT = 'test.txt'

-- make dataset
local function make_tok(fnTxt)
    local tcmd = {"perl",
        PL_SCRIPT, -- program
        fnTxt, FN_REGISTER, '.tok', -- args
    }
    local cmd = table.concat(tcmd, ' ')

    print('running command:')
    print(cmd)
    os.execute(cmd)
end

local function main()
    local fnTxt = path.join(PATH_DATA, FN_TXT)
    print('input txt: ' .. fnTxt)
    make_tok(fnTxt)
end

main()