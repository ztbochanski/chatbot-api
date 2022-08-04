import pandas as pd


class Predictor:

    def __init__(self, predictions=None) -> None:
        self.predictions = predictions

    def predict(self, model, input):
        df_input = pd.DataFrame([input],
                                dtype=float, index=['user_input'])
        predictions = model.predict(df_input)
        self.predictions = predictions.flatten().tolist()
        print(self.predictions)

    def filter_predictions(self, buffer):
        self.predictions = [[label, pred]
                            for label, pred in enumerate(self.predictions) if pred > buffer]

    def sort_predictions(self):
        self.predictions.sort(key=lambda x: x[1], reverse=True)

    def build_hash(self, labels):
        prediction_hash = []
        for prediction in self.predictions:
            prediction_hash.append(
                {'intent': labels[prediction[0]], 'probability': str(prediction[1])})
        return prediction_hash
