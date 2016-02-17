# -*- coding: utf-8 -*-
import json
import re
import gensim

__author__ = 'yixuanhe'


# poem exact, deal with some noise in poem data
def data_extract(data_location):
    poems = []
    ch = {}
    i = 1
    num = 0
    with open(data_location) as f:
        for l in f:
            flag = False
            num += 1

            middle = []
            obj = json.loads(l)
            s = obj['result']['content']
            s = s.replace(" ", "")

            # deal with [XXXXXX]
            s = re.sub(r'\[[^\[]*\]', '', s)
            # deal with 《XXXX》
            s = re.sub(u'《[^《]*》', '', s)

            # deal with "?"in poem
            if u"？" in s:
                lines = s.split(u"？")
                for line in lines:
                    if line != '':
                        line += u"？"
                        middle.append(line)
            else:
                middle.append(s)

            for m in middle:
                lines = m.split(u"。")
                for line in lines:
                    if line != '':
                        if not line.endswith(u"？"):
                            line += u"。"
                        if len(line) == 12 or len(line) == 16:
                            leng = len(line)
                            for c in line:
                                if c != " " and not (c in ch.keys()):
                                    ch[c] = i
                                    i += 1
                            for i in range(leng):
                                pos = leng - i - 1
                                line = line[:pos] + " " + line[pos:]
                            poems.append(line)
                        flag = True
    return poems, ch


# training vord2vec using data
def train(poems):
    num_features = 300
    min_word_count = 0
    num_workers = 48
    context = 20
    epoch = 20
    sample = 1e-5
    model = gensim.models.word2vec.Word2Vec(
        poems,
        size=num_features,
        workers=num_workers,
        min_count=min_word_count,
        sample=sample,
        window=context,
        iter=epoch,
    )
    return model

# generate train data
def genTrainData(poems, model, data_location, ch_set):
    with open(data_location, "w") as f:
        for poem in poems:
            leng = len(poem)
            for i in range(leng):
                if poem[i] != " ":
                    features = model[poem[i]].tolist()
                    for feature in features:
                        f.write(str(feature))
                        f.write(",")
                    f.write(",")
                    if leng == 12:
                        f.write("1")
                    else:
                        f.write("0")
                    f.write(" ")
                    if i+2 > leng:
                        f.write("-1")
                    else:
                        f.write(str(ch_set[poem[i]]))

                    f.write("\n")


poems, ch_set = data_extract("data/test.json")
print(len(poems))
print(len(ch_set))
with open("data/ch_map", "w") as f:
    for k in ch_set:
        f.write(k + "," + str(ch_set[k]) + "\n")

print("********** !data extract finish!************")
model = train(poems)
model.save("data/model")
print("********** !model training finish!************")
genTrainData(poems, model, "/Volumes/devil/train_data", ch_set)
print("********** !write to file finish!************")
