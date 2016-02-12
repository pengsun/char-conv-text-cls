--- make imdb dataset, use pre-defined lower-case char vocab
require'pl.path'
require'pl.stringx'
require'pl.file'

--- global config: dbpedia train
local PATH_CSV = '/home/ps/data/dbpedia'
local FN_CSV = 'train.csv'
local PATH_TXT_CAT = '/home/ps/data/dbpedia/tok-cat'
local FN_TXT = 'train.txt'
local FN_CAT = 'train.cat'

--- global config: dbpedia test
local PATH_CSV = '/home/ps/data/dbpedia'
local FN_CSV = 'test.csv'
local PATH_TXT_CAT = '/home/ps/data/dbpedia/tok-cat'
local FN_TXT = 'test.txt'
local FN_CAT = 'test.cat'


--- parse CSV line
-- from: Zhang Xiang's Crepe
-- Reference: http://lua-users.org/wiki/LuaCsv
local function parse_csv_line (line,sep)
    local res = {}
    local pos = 1
    sep = sep or ','
    while true do
        local c = string.sub(line,pos,pos)
        if (c == "") then break end
        if (c == '"') then
            -- quoted value (ignore separator within)
            local txt = ""
            repeat
                local startp,endp = string.find(line,'^%b""',pos)
                txt = txt..string.sub(line,startp+1,endp-1)
                pos = endp + 1
                c = string.sub(line,pos,pos)
                if (c == '"') then txt = txt..'"' end
                -- check first char AFTER quoted string, if it is another
                -- quoted string without separator, then append it
                -- this is the way to "escape" the quote char in a quote. example:
                --   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
            until (c ~= '"')
            table.insert(res,txt)
            assert(c == sep or c == "")
            pos = pos + 1
        else
            -- no quotes used, just look for the first separator
            local startp,endp = string.find(line,sep,pos)
            if (startp) then
                table.insert(res,string.sub(line,pos,startp-1))
                pos = endp + 1
            else
                -- no separator found -> use rest of string and terminate
                table.insert(res,string.sub(line,pos))
                break
            end
        end
    end
    return res
end

-- make dataset
local function make_txt_cat(fnCSV, fnTxt, fnCat)
    local sep = "," -- seperated by comma
    local iCat = 1 -- CSV column 1: category
    local iTok = 3 -- CSV column 3: tokens

    local fCSV = assert(io.open(fnCSV, "r"), "cannot open " .. fnCSV)
    local fTok = assert(io.open(fnTxt, "w"), "cannot open " .. fnTxt)
    local fCat = assert(io.open(fnCat, "w"), "cannot open " .. fnCat)

    -- processing line by line
    local n = 0
    for line in fCSV:lines() do
        n = n + 1
        if n % 1000 == 0 then
            io.write("\rprocessing line " .. n)
            io.flush()
        end

        local items = parse_csv_line(line, ",")
        assert(#items==3, fnCSV .. ": corrupted line: " .. n)

        -- write line: texts
        local toks = items[iTok]
        fTok:write(toks .. "\n")

        -- write line: the category
        local cat = items[iCat]
        fCat:write(cat .. "\n")
    end
    print('\ndone. #lines processed: ' .. n .. "\n")

    fCSV:close()
    fTok:close()
    fCat:close()
end

local function main()
    local fnCSV = path.join(PATH_CSV, FN_CSV)
    local fnTxt = path.join(PATH_TXT_CAT, FN_TXT)
    local fnCat = path.join(PATH_TXT_CAT, FN_CAT)
    print('input CSV: ' .. fnCSV)
    print('saving tokens to: ' .. fnTxt)
    print('saving category to: ' .. fnCat)
    make_txt_cat(fnCSV, fnTxt,fnCat)
end

main()