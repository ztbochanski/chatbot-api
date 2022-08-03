from ml_api.input_parser import Encoder
import pandas as pd
import pickle


def test_model():
    with open(f'./training/training_data/surgery-recovery-assistant-data.pkl', 'rb') as data_file:
        data = pickle.load(data_file)
        stemmed_words = data['stemmed_words']
        labels = data['labels']

    with open(f'./training/training_data/surgery-recovery-assistant-model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    print('Pharmacy Search is user intent')
    test_user_input = 'Where can I get my medications?'

    encoder = Encoder()
    encoded_user_input = encoder.encode_input(test_user_input, stemmed_words)
    print('Encoded input:', encoded_user_input)
    print('Intent labels:', labels)

    df_input = pd.DataFrame([encoded_user_input],
                            dtype=float, index=['user_input'])
    predictions = model.predict(df_input)
    preds = predictions.flatten().tolist()

    for label in range(len(labels)):
        print('Predicted intent: ' +
              '{:.06%}'.format(preds[label]) + ' \'' + labels[label] + '\'')

    if preds[4] > .9:
        print('PASSED!')
        print(preds[4], '-> Pharmacy search was correctly predicted')
    else:
        print('FAILED!')
        print('Pharmacy search not predicted')


if __name__ == '__main__':
    test_model()
