import numpy as np

def forecast_lstm_8weeks(model, X_input):

    preds = []
    current = list(X_input)

    for _ in range(8):

        x = np.array(current).reshape((1,1,len(current)))
        pred = model.predict(x)[0][0]

        preds.append(pred)

        current = current[1:] + [pred]

    return preds