--- CSV to text and category
require'pl.path'
require'pl.stringx'
require'pl.file'

local PL_SCRIPT = path.join(
    'data-prep', 'convert.pl'
)

-- helper: make dataset
local function make_txt_cat(inputName, outputName)
    local tcmd = {
        "perl",
        PL_SCRIPT, -- program
        inputName,
        outputName,
    }
    local cmd = table.concat(tcmd, ' ')

    print('running command: ')
    print(cmd)
    local code = os.execute(cmd)
    print("\n")
end

-- interface
local this = {}

this.main = function(opt)

    --- default/examplar option
    local opt = opt or {
        -- input
        input_path = '/mnt/data/datasets/Text/dbpedia/test.csv',
        -- output
        output_path_name = '/mnt/data/datasets/Text/dbpedia/tok-cat/test',
    }

    print('input CSV: ' .. opt.input_path)
    print('saving tokens, category to: ' .. opt.output_path_name)
    make_txt_cat(opt.input_path, opt.output_path_name)
end

return this