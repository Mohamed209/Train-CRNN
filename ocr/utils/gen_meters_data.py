from random import randint
savepth = '../../dataset/text_corpus/'
num_map = {0: '٠', 1: '١', 2: '٢', 3: '٣', 4: '٤',
           5: '٥', 6: '٦', 7: '٧', 8: '٨', 9: '٩'}
with open(savepth+'arameters.txt', mode='w', encoding='utf-8') as arafile:
    for j in range(2000):
        samples = []
        slen = randint(4, 8)
        for i in range(slen):
            samples.append(num_map[randint(0, 9)])
        arafile.writelines([i for i in samples])
        arafile.write('\n')
with open(savepth+'engmeters.txt', mode='w', encoding='utf-8') as engfile:
    for j in range(2000):
        samples = []
        slen = randint(4, 8)
        for i in range(slen):
            samples.append(randint(0, 9))
        engfile.writelines([str(i) for i in samples])
        engfile.write('\n')
