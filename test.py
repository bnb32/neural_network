from model import predict_prob
import keras

model = keras.models.load_model('./data/nn_model')

texts = [
         "hello",
         "LOL finger me with battery acid ROFLMAO. ",
         "KKomrade ti si glup kokurac. ",
         "fuck you",
         "fuck you bitch",
         "youre such a simp",
         "just chatting",
        ]

print(model.predict(texts))

print(predict_prob(model,texts))
