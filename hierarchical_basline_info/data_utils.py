import json

## handle headacheboy (this directory) outputs ##

def write_labels():

    f_test = open("text/test2.txt", "r")

    f_wrong = open("output/wrongList", "r")
    wrongList = f_wrong.readlines()[0][1:-1].split(', ')
    f_wrong.close()

    d_wrong = {}
    d_correct = {}
    for line in f_test.readlines():
        line = eval(line.strip())
        tweet_id = line[0]
        caption = line[1]
        # hash_label = int(line[-6])

        label = int(line[-2]) # annotator label # this is what the other paper uses

        if tweet_id in wrongList:
            if label == 0:
                prediction = 1
            else:
                prediction = 0    
            d_wrong[tweet_id] = {
                    "pred": prediction,
                    "label": label,
                    "caption": caption
                    }
        else:
            d_correct[tweet_id] = {
                    "label": label,
                    "caption": caption
                    }
        
    f_test.close()
 
        
    f_out = open("output/headache_wrongDict.json", "w")
    json.dump(d_wrong, f_out, indent=4)
    f_out.close()

    f_out = open("output/headache_correctDict.json", "w")
    json.dump(d_correct, f_out, indent=4)
    f_out.close()

def get_wrong():

    f = open("output/headache_wrongDict.json", "r")
    data = json.load(f)
    f.close()

    print("headache wrong: ", len(data))
    headache_fp = {k:v for k,v in data.items() if v["label"] == 0}
    headache_fn = {k:v for k,v in data.items() if v["label"] == 1}
    print("false positives: ", len(headache_fp))
    print("false negatives: ", len(headache_fn))

    return data

def get_correct():

    f = open("output/headache_correctDict.json", "r")
    data = json.load(f)
    f.close()

    print("headache correct: ", len(data))
    print("true positives: ", len([k for k,v in data.items() if v["label"] == 1]))
    print("true negatives: ", len([k for k,v in data.items() if v["label"] == 0]))

    return data

### handle MSDBert (downloaded from aneesha) outputs ##

import csv

def reformat_msd():

    f = open("MsdBERT.csv", "r")
    csv_reader = csv.reader(f, delimiter=',')

    d_correct = {}
    d_wrong = {}
    for row in csv_reader:
        tweet_id = eval(row[0])[0]
        caption = eval(row[0])[1]
        label = int(row[2])
        correct = int(row[-1])

        if correct == 1:
            d_correct[tweet_id] = {
                    "label": label,
                    "caption": caption
                    }
        else:
            d_wrong[tweet_id] = {
                    "prediction": row[1],
                    "label": label,
                    "caption": caption
                    }
    
    f_out = open("output/msdBERT_wrong.json", "w")
    json.dump(d_wrong, f_out, indent=4)
    f_out.close()

    print("msd wrong: ", len(d_wrong))

    f_out = open("output/msdBERT_correct.json", "w")
    json.dump(d_correct, f_out, indent=4)
    f_out.close()

    print("msd correct: ", len(d_correct))

def get_msd_wrong():

    f = open("output/msdBERT_wrong.json", "r")
    data = json.load(f)
    f.close()

    return data

def get_msd_correct():

    f = open("output/msdBERT_correct.json", "r")
    data = json.load(f)
    f.close()

    print("msd correct: ", len(data))
    print("true positives: ", len([k for k,v in data.items() if v["label"] == 1]))
    print("true negatives: ", len([k for k,v in data.items() if v["label"] == 0]))

    return data

### error analysis ###

def get_venn_diagram(headache_wrong, msd_wrong, headache_correct, msd_correct):
    
    wrong_intersection = {k:v for k,v in headache_wrong.items() if k in msd_wrong}
 
    # f_out = open("output/wrongIntersection.json", "w")
    # json.dump(wrong_intersection, f_out, indent=4)
    # f_out.close()  

    print("wrong intersection: ", len(wrong_intersection))
    intersection_fp = {k:v for k,v in wrong_intersection.items() if v["label"] == 0}
    intersection_fn = {k:v for k,v in wrong_intersection.items() if v["label"] == 1}
    print("false positives: ", len(intersection_fp))
    print("false negatives: ", len(intersection_fn))

    correct_intersection = {k:v for k,v in headache_correct.items() if k in msd_correct}
    print("correct intersection: ", len(correct_intersection))
    intersection_fp = {k:v for k,v in correct_intersection.items() if v["label"] == 0}
    intersection_fn = {k:v for k,v in correct_intersection.items() if v["label"] == 1}

    # f_out = open("output/intersection_fp.json", "w")
    # json.dump(intersection_fp, f_out, indent=4)
    # f_out.close()
    # f_out = open("output/intersection_fn.json", "w")
    # json.dump(intersection_fn, f_out, indent=4)
    # f_out.close()

    headache_wrong_only = {k:v for k,v in headache_wrong.items() if k not in msd_wrong}
    msd_wrong_only = {k:v for k,v in msd_wrong.items() if k not in headache_wrong}

    # f_out = open("output/headache_wrongOnly.json", "w")
    # json.dump(headache_wrong_only, f_out, indent=4)
    # f_out.close()

    print("headache wrong only: ", len(headache_wrong_only))
    headache_fp = {k:v for k,v in headache_wrong_only.items() if v["label"] == 0}
    headache_fn = {k:v for k,v in headache_wrong_only.items() if v["label"] == 1}
    print("false positives: ", len(headache_fp))
    print("false negatives: ", len(headache_fn))

    # f_out = open("output/headache_only_fp.json", "w")
    # json.dump(headache_fp, f_out, indent=4)
    # f_out.close()
    # f_out = open("output/headache_only_fn.json", "w")
    # json.dump(headache_fn, f_out, indent=4)
    # f_out.close()

    # f_out = open("output/msdWrongOnly.json", "w")
    # json.dump(msd_wrong_only, f_out, indent=4)
    # f_out.close()

    print("msd wrong only: ", len(msd_wrong_only))
    msd_fp = {k:v for k,v in msd_wrong_only.items() if v["label"] == 0}
    msd_fn = {k:v for k,v in msd_wrong_only.items() if v["label"] == 1}
    print("false positives: ", len(msd_fp))
    print("false negatives: ", len(msd_fn))

    # f_out = open("output/msd_only_fp.json", "w")
    # json.dump(msd_fp, f_out, indent=4)
    # f_out.close()
    # f_out = open("output/msd_only_fn.json", "w")
    # json.dump(msd_fn, f_out, indent=4)
    # f_out.close()

    headache_correct_only = {k:v for k,v in headache_correct.items() if k not in msd_correct}
    msd_correct_only = {k:v for k,v in msd_correct.items() if k not in headache_correct}
    print("headache correct only: ", len(headache_correct_only))
    print("msd correct only: ", len(msd_correct_only))

import shutil
def get_wrong_images():
    
    f = open("output/wrongList", "r")
    headache_wrong = eval(f.readlines()[0])
    f.close()

    f = open("output/msdBERT_wrong.json", "r")
    msd_wrong = [k for k,v in json.load(f).items()]
    f.close()

    wrong = list(set(headache_wrong + msd_wrong))
    for tweet_id in wrong:
        fname = str(tweet_id)
        shutil.copy("dataset_image/"+fname+".jpg", "wrong_images")

def get_correct_images(): #some at least

    f = open("output/headache_correctDict.json", "r")
    data = json.load(f)
    f.close()

    for k,v in data.items()[:20]:
        shutil.copy("dataset_image/"+k+".jpg", "some_correct_images")
        

# # write_labels()
headache_wrong = get_wrong()
headache_correct = get_correct()
# reformat_msd()
msd_wrong = get_msd_wrong()
msd_correct = get_msd_correct()
get_venn_diagram(headache_wrong, msd_wrong, headache_correct, msd_correct)
# # get_wrong_images()
# # get_correct_images()
