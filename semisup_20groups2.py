from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data["data"]
categories = data["target"]
category_names = data["target_names"]

labels_to_add = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x',]
indices = [category_names.index(label) for label in labels_to_add]
y = [label if label in indices else -1 for label in categories]

topic_model = BERTopic(verbose=True).fit(docs, y=y)
print(topic_model)
