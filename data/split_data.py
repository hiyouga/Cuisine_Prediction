import json
import random


if __name__ == '__main__':
    train = json.load(open('train_raw.json', 'r', encoding='utf-8'))
    dev_ratio = 0.05
    length_dict_1 = dict()
    length_dict_2 = dict()
    for data in train.values():
        length_1 = len(data['ingredients'])
        length_2 = len((' '.join(data['ingredients'])).split())
        if length_1 in length_dict_1:
            length_dict_1[length_1] += 1
        else:
            length_dict_1[length_1] = 1
        if length_2 in length_dict_2:
            length_dict_2[length_2] += 1
        else:
            length_dict_2[length_2] = 1
    ids = list(train.keys())
    random.shuffle(ids)
    train_ids, dev_ids = ids[:int(len(ids) * (1-dev_ratio))], ids[int(len(ids) * (1-dev_ratio)):]
    trainset, devset = dict(), dict()
    for k, w in train.items():
        if k in train_ids:
            trainset[k] = w
        else:
            devset[k] = w
    with open('train.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(trainset, sort_keys=False, indent=4))
        print(f"Processed {len(trainset)} training examples")
    with open('dev.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(devset, sort_keys=False, indent=4))
        print(f"Processed {len(devset)} development examples")
    testset = json.load(open('test_raw.json', 'r', encoding='utf-8'))
    with open('test.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(testset, sort_keys=False, indent=4))
        print(f"Processed {len(testset)} test examples")
