import json

import pandas as pd
import tensorflow as tf
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    print('Reading csv')
    data = pd.read_csv("test.csv")
    idxs = data["idx_test"].tolist()
    paths = data["path_img"].tolist()

    # Load the model
    cnn = tf.keras.models.load_model("saved_models/sparse", compile=False)
    cnn.compile()

    sizeimage = 100

    results = {
        "target": {}
    }
    for i, path in enumerate(paths):
        img = tf.keras.utils.load_img(
            path, target_size=(sizeimage, sizeimage)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = cnn.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        results['target'][idxs[i]] = np.argmax(score)

        if i % 100 == 0:
            print(f"Prediction at {float(i) / len(paths) * 100} %")

    # Write to json
    with open("results_sparse.json", "w") as outfile:
        json.dump(results, outfile, cls=NpEncoder)
