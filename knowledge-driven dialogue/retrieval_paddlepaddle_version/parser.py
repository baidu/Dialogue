import json
import sys
import re


def parser_json(line): 
    json_data = json.loads(line)
    return json_data


def parser_out(line): 
    json_data = parser_json(line)

    chat_path = []
    for path in json_data['chat_path']: 
        path_str = "\1".join(path)
        chat_path.append(path_str.encode('utf8'))
    chat_strs = "\t".join(chat_path)

    spo_list = []
    for spo in json_data['knowledge']:
        spo_str = "\1".join(spo)
        spo_list.append(spo_str.encode('utf8'))
    spo_strs = "\t".join(spo_list)
    
    if json_data['history']: 
        his_str = "\t".join(json_data['history']).encode('utf8')
    else: 
        his_str = "[START]"
    
    response_list = []
    label_list = []
    for res in json_data['candidate']: 
        response_list.append(res[0].encode('utf8'))
        label_list.append(str(res[1]).encode('utf8'))
    response_str = "\t".join(response_list)
    chat_elems = [chat_strs, spo_strs, his_str, response_str, label_list]
    out_list = parser_char(chat_elems)
    return out_list


def check_elem(txt): 
    if txt.isdigit(): 
        return txt
    txt = txt.decode('utf8')
    for i in range(len(txt)): 
        if txt[i] >= u'\u4e00' and txt[i] <= u'\u9fa5': 
            txt_out = " ".join([t.encode('utf8') for t in txt])
            txt_out = re.sub(" +", " ", txt_out)
            return txt_out
    return txt.encode('utf8')


def parser_part(line, tag): 
    line = re.sub('\1', ' ', line)
    line = re.sub(' +', ' ', line)
    elems = line.split('\t')
    out_list = []
    for elem in elems: 
        ws = elem.split()
        ws_list = []
        for w in ws: 
            split_str = check_elem(w)
            ws_list.append(split_str)
        out_list.append(" ".join(ws_list))
    return (tag).join(out_list)


def parser_char(chat_elems): 
    chat_path = chat_elems[0]
    kn_str = chat_elems[1]
    history = chat_elems[2]
    response = chat_elems[3]
    labels = chat_elems[4]

    chat_path_out = parser_part(chat_path, ' [PATH_SEP] ')
    chat_path_out = re.sub(" +", " ", chat_path_out)

    kn_str_out = parser_part(kn_str, ' [KN_SEP] ')
    kn_str_out = re.sub(" +", " ", kn_str_out)

    history_out = parser_part(history, ' [INNER_SEP] ')
    history_out = re.sub(" +", " ", history_out)

    response_out = parser_part(response, '\t')
    response_list = response_out.split('\t')

    out_list = []
    for i in range(len(response_list)): 
        out = [labels[i], history_out, response_list[i], chat_path_out, kn_str_out]
        out_list.append(out)
    return out_list


def dump_json(json_file, score_out): 
    idx = 0
    for line in open(json_file):
        line = line.strip()
        json_data = json.loads(line)
        for elem in json_data['candidate']:
            elem.append(score_out[idx])
            idx += 1
            out = json.dumps(json_data, ensure_ascii=False)
        print out.encode('utf8')


if __name__ == "__main__": 
    
    for line in sys.stdin: 
        line = line.rstrip('\n')
        out_list = parser_out(line)
        print "\n".join(out_list)




