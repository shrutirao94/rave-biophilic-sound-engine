import lmdb
from udls.generated import AudioExample

env = lmdb.open("data/preprocessed_rave/data_classes/birds", lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    for k, v in cursor:
        ae = AudioExample.FromString(v)
        if 'length' not in ae.metadata or ae.metadata['length'] == '':
            print(f"Missing length in key: {k}")

