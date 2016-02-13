--- make imdb dataset, use pre-defined lower-case char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

--- internal config: pl script and registerd-token
local FN_REGISTER = path.join(
    'util', 'data',
    'registered-tokens.txt'
)
local PL_SCRIPT = path.join(
    'util', 'data',
    'to_tokens.pl'
)

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
    print("\n")
end

--
local this = {}

this.main = function(opt)
    --- default/examplar option
    local opt = opt or {
        -- input
        path_data = '/mnt/data/datasets/Text/dbpedia/tok-cat',
        fn_txt = 'test.txt',
    }

    local fnTxt = path.join(opt.path_data, opt.fn_txt)
    print('input txt: ' .. fnTxt)
    make_tok(fnTxt)
end

return this