from time import time

from summa import keywords
from summa import summarizer

with open('./data/clean-1905.03298.txt', 'r') as input_fp:
    lines = input_fp.read().splitlines()

text = ' '.join(lines)

for ratio in range(1, 4):
    float_ratio = float(ratio) / 10.0
    time_before = time()
    summary = summarizer.summarize(ratio=float_ratio, text=text)
    sentences = summary.split('.')
    sentences = [item.replace('\n', '') + '.' for item in sentences if len(item) > 0]
    for item in sentences:
        print('{} : {}'.format(len(item), item))
    time_after = time()
    print('summarize takes {:5.4f}s for ratio {:5.2f} and summary has length {}'.format(time_after - time_before,
                                                                                        float_ratio, len(summary)))

print('keywords: {}'.format(keywords.keywords(text=text)))

quit(0)
