from keras.models import Sequential
from keras.layers import InputLayer, Bidirectional, LSTM, Dropout, Dense, Flatten

def is_last_layer(stack_id, num_of_stack):
    if stack_id == num_of_stack-1:
        return True
    else:
        return False

def create_rnn_model(input_shape=(30,7), 
                    units=[128, 64], 
                    dropout=0.25):
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    
    num_of_stack = len(units)
    
    for i in range(num_of_stack):
        if is_last_layer(i, num_of_stack):
            model.add(Bidirectional(LSTM(units=units[i], return_sequences=False)))
            model.add(Dropout(dropout))
            continue
        model.add(Bidirectional(LSTM(units=units[i], return_sequences=True)))
    
    model.add(Dense(units=60, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(units=1, activation='linear'))

    return model

# 사용 예시
if __name__ == '__main__':
    model = create_rnn_model(units=[128, 64])
    model.summary()